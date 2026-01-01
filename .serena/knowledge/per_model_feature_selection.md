# Per-Model Feature Selection Architecture

## Core Principle

**Each model family gets a DIFFERENT feature set** tailored to its inductive biases and learning capabilities.

**Key Points:**
- Same underlying 1-min canonical OHLCV source
- Same timestamps, labels, splits
- DIFFERENT primary timeframes per model
- DIFFERENT feature sets per model
- DIFFERENT MTF strategies per model

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│              1-MIN CANONICAL OHLCV (SINGLE SOURCE)                  │
│                                                                     │
│  data/raw/MES_1m.parquet                                           │
│  - Single timestamp index                                          │
│  - OHLCV columns + volume                                          │
│  - No gaps (preserved, not filled)                                 │
└────────────────────────┬────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────────────┐
│              MTF UPSCALING (9 TIMEFRAMES)                           │
│                                                                     │
│  1min → 5min → 10min → 15min → 20min → 25min → 30min → 45min → 1h │
│  All aligned to same timestamps (forward-fill + shift(1))         │
└────────────────────────┬────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────────────┐
│              PER-MODEL FEATURE SELECTION                            │
│                                                                     │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐           │
│  │   CatBoost   │   │     TCN      │   │   PatchTST   │           │
│  │  (Tabular)   │   │ (Sequence)   │   │ (Transformer)│           │
│  ├──────────────┤   ├──────────────┤   ├──────────────┤           │
│  │ Primary: 15m │   │ Primary: 5m  │   │ Primary: 1m  │           │
│  │ MTF: Yes     │   │ MTF: No      │   │ MTF: 3 stream│           │
│  │ Features:    │   │ Features:    │   │ Features:    │           │
│  │  ~200 eng    │   │  ~150 base   │   │  Raw OHLCV   │           │
│  └──────┬───────┘   └──────┬───────┘   └──────┬───────┘           │
│         ↓                  ↓                  ↓                     │
│    (N, 200)           (N, 60, 150)       (N, 3, 60, 4)             │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Feature Selection by Model Family

### Tabular Models (Boosting + Classical)

**Models:** XGBoost, LightGBM, CatBoost, Random Forest, Logistic, SVM

**Feature Requirements:**
- Engineered, stationary-ish indicators
- MTF indicators for cross-timeframe relationships
- Higher primary timeframe (15min or 1h) to reduce noise

**Example - CatBoost:**
- **Primary TF:** 15min (resampled from 1-min canonical)
- **MTF Strategy:** MTF Indicators (Strategy 2)
- **Feature Set:**
  - Base indicators on 15min: ~60 features
    - RSI (14, 21, 28)
    - MACD (12-26-9, 19-39-9)
    - ATR (14, 21)
    - Bollinger Bands (20, 2.0)
    - ADX (14)
    - Stochastic (14, 3, 3)
  - Wavelets on 15min: ~24 features
    - Db4 decomposition (3 levels)
    - Haar decomposition (3 levels)
  - Microstructure proxies on 15min: ~10 features
    - Roll spread
    - Kyle's lambda
    - Amihud illiquidity
    - Effective spread
  - MTF indicators from 1m/5m/1h: ~50 features
    - RSI_1m, RSI_5m, RSI_1h
    - MACD_1m, MACD_5m, MACD_1h
    - ATR_1m, ATR_5m, ATR_1h
    - Volume ratios across timeframes
  - Price/volume features: ~8 features
  - Time features (hour, day_of_week): ~2 features
  - **Total: ~200 features (2D array)**

**Example - Random Forest:**
- **Primary TF:** 1h (resampled from 1-min canonical)
- **MTF Strategy:** Single-TF (no MTF enrichment)
- **Feature Set:**
  - Base indicators on 1h: ~60 features
  - Wavelets on 1h: ~24 features
  - Microstructure on 1h: ~10 features
  - Price/volume features: ~8 features
  - Time features: ~2 features
  - **Total: ~150 features (2D array)**
  - No MTF indicators (higher TF already smooths noise)

**Why Different Primary TFs?**
- CatBoost (15min): Balance between noise reduction and pattern detail
- Random Forest (1h): Simpler decision boundaries prefer smoother data

---

### Sequence Models (Neural + CNN)

**Models:** LSTM, GRU, TCN, Transformer, InceptionTime, 1D ResNet

**Feature Requirements:**
- Raw or lightly processed features
- Temporal context (3D windows)
- Lower primary timeframe (5min) for local patterns
- Single-TF or limited MTF (model learns temporal relationships)

**Example - TCN:**
- **Primary TF:** 5min (resampled from 1-min canonical)
- **MTF Strategy:** Single-TF (no MTF enrichment)
- **Feature Set:**
  - Base indicators on 5min: ~60 features
    - RSI (14, 21, 28)
    - MACD (12-26-9, 19-39-9)
    - ATR (14, 21)
    - Bollinger Bands (20, 2.0)
    - ADX (14)
  - Wavelets on 5min: ~24 features
    - Db4 decomposition (3 levels)
  - Microstructure on 5min: ~10 features
    - Roll spread
    - Volume imbalance
  - Price/volume raw features: ~8 features
    - Returns (1, 5, 10 bar)
    - Log volume
    - OHLC ratios
  - Time features: ~2 features
  - **Total: ~150 features in 3D windows**
  - **Shape:** (N, seq_len=60, 150)
  - No MTF indicators (TCN learns from temporal sequence)

**Example - LSTM:**
- **Primary TF:** 5min (resampled from 1-min canonical)
- **MTF Strategy:** Single-TF (no MTF enrichment)
- **Feature Set:**
  - Same as TCN (~150 features)
  - **Shape:** (N, seq_len=30, 150) - shorter lookback for LSTM memory

**Why Single-TF for Sequence Models?**
- Sequence models have inherent temporal memory
- Adding MTF indicators is redundant (model learns multi-scale patterns from sequence)
- Cleaner signal (no feature engineering, let model learn)

---

### Advanced Transformers (Multi-Resolution)

**Models:** PatchTST, iTransformer, TFT, N-BEATS

**Feature Requirements:**
- Multi-resolution raw OHLCV bars
- Multiple timeframe streams as separate channels
- Lowest primary timeframe (1min) for maximum information
- MTF Ingestion strategy (Strategy 3)

**Example - PatchTST:**
- **Primary TF:** 1min (canonical, no resampling)
- **MTF Strategy:** MTF Ingestion (multi-stream, Strategy 3)
- **Feature Set:**
  - **Stream 1:** 1min raw OHLCV (60 bars × 4 OHLC)
  - **Stream 2:** 5min raw OHLCV (60 bars × 4 OHLC)
  - **Stream 3:** 15min raw OHLCV (60 bars × 4 OHLC)
  - **Total: 3 streams × 60 bars × 4 OHLC = 4D tensor (N, 3, 60, 4)**
  - Model learns cross-timeframe patterns via patch-based attention

**Example - TFT (Temporal Fusion Transformer):**
- **Primary TF:** 1min (canonical, no resampling)
- **MTF Strategy:** MTF Ingestion (multi-stream + context, Strategy 3)
- **Feature Set:**
  - **Temporal inputs (3D):**
    - Stream 1: 1min raw OHLCV (60, 4)
    - Stream 2: 5min raw OHLCV (60, 4)
    - Stream 3: 15min raw OHLCV (60, 4)
  - **Static context (1D):**
    - Regime features (volatility, trend): ~4 features
    - Market session: ~1 feature
  - **Total: 4D temporal (N, 3, 60, 4) + 1D static (N, 5)**
  - Model fuses temporal streams with static context via variable selection

**Why Raw OHLCV for Transformers?**
- Attention mechanisms excel at learning from raw data
- Pre-engineered indicators limit what model can learn
- Multi-resolution inputs enable hierarchical pattern learning

---

### Inference/Meta Models

**Models:** Logistic Meta, Ridge Meta, MLP Meta, Calibrated Blender

**Feature Requirements:**
- OOF predictions from heterogeneous base models
- Optional lightweight context features
- No primary timeframe (operates on predictions, not raw data)

**Example - Logistic Meta:**
- **Input:** OOF predictions from CatBoost + TCN + PatchTST
- **Feature Set:**
  - CatBoost predictions (3 classes): 3 features
  - TCN predictions (3 classes): 3 features
  - PatchTST predictions (3 classes): 3 features
  - Optional context: regime (1), volatility (1)
  - **Total: ~11 features (2D array)**

**Example - MLP Meta (with context):**
- **Input:** OOF predictions + market context
- **Feature Set:**
  - Base model predictions: 9 features (3 models × 3 classes)
  - Regime features: 4 features (volatility regime, trend regime, composite regime, strength)
  - Time features: 2 features (hour, day_of_week)
  - Recent volatility: 1 feature (rolling 20-bar ATR)
  - **Total: ~16 features (2D array)**
  - **Architecture:** 16 → 32 → 16 → 3 (2 hidden layers)

**Why Predictions as Features?**
- Meta-learner blends model outputs, not raw data
- Each base model has already processed raw data via its optimal feature set
- Meta-learner learns which model to trust when (model selection)

---

## Why Different Features Matter

### Example: Heterogeneous Ensemble

**Scenario:** Train CatBoost + TCN + PatchTST → Logistic Meta-Learner

| Model | Primary TF | MTF Strategy | Features | Input Shape | Why Different? |
|-------|-----------|--------------|----------|-------------|----------------|
| **CatBoost** | 15min | MTF Indicators | ~200 engineered | (N, 200) | Boosting excels with rich feature interactions across timeframes |
| **TCN** | 5min | Single-TF | ~150 in 3D | (N, 60, 150) | CNN captures local temporal patterns, no need for pre-engineered MTF |
| **PatchTST** | 1min | MTF Ingestion | 3 streams × 4 OHLC | (N, 3, 60, 4) | Transformer learns multi-resolution attention, needs raw data |

**Result:**
- Each model learns from different aspects of the same underlying 1-min OHLCV
- Diversity of feature representations → reduced error correlation
- Meta-learner combines complementary predictions

---

### Diversity Mechanisms

**1. Primary Timeframe Diversity:**
- CatBoost (15min): Pattern recognition with noise reduction
- TCN (5min): Local temporal patterns with moderate noise
- PatchTST (1min): Maximum information, model handles noise

**2. Feature Engineering Diversity:**
- CatBoost: Heavily engineered (RSI, MACD, wavelets, MTF indicators)
- TCN: Lightly engineered (basic indicators + raw features)
- PatchTST: Raw OHLCV bars (no engineering, model learns)

**3. MTF Strategy Diversity:**
- CatBoost: MTF Indicators (cross-timeframe features as columns)
- TCN: Single-TF (temporal sequence provides scale)
- PatchTST: MTF Ingestion (multi-stream raw data as channels)

**4. Temporal Context Diversity:**
- CatBoost: 1 bar (instant features)
- TCN: 60 bars (local patterns)
- PatchTST: 60 bars × 3 streams (hierarchical patterns)

---

## Implementation

### Per-Model Configuration

**Config Structure:** `config/models/{model_name}.yaml`

**Example - CatBoost:**
```yaml
# config/models/catboost.yaml
model_type: "tabular"
primary_timeframe: "15min"
mtf_strategy: "mtf_indicators"
mtf_sources: ["1min", "5min", "1h"]
feature_groups:
  - "indicators"
  - "wavelets"
  - "microstructure"
  - "mtf_indicators"
  - "price_volume"
  - "time"

model_params:
  iterations: 1000
  depth: 6
  learning_rate: 0.05
```

**Example - TCN:**
```yaml
# config/models/tcn.yaml
model_type: "sequence"
primary_timeframe: "5min"
mtf_strategy: "single_tf"
sequence_length: 60
feature_groups:
  - "indicators"
  - "wavelets"
  - "microstructure"
  - "price_volume"
  - "time"

model_params:
  num_channels: [128, 128, 128]
  kernel_size: 3
  dropout: 0.2
```

**Example - PatchTST:**
```yaml
# config/models/patchtst.yaml
model_type: "multi_resolution"
primary_timeframe: "1min"
mtf_strategy: "mtf_ingestion"
mtf_streams: ["1min", "5min", "15min"]
sequence_length: 60
feature_groups:
  - "raw_ohlcv"

model_params:
  patch_len: 16
  stride: 8
  d_model: 128
  n_heads: 8
  n_layers: 3
  dropout: 0.1
```

---

### Feature Selection Logic

**Location:** `src/phase5/adapters/feature_selector.py`

```python
from typing import Dict, List
import pandas as pd
import numpy as np

class FeatureSelector:
    """Select features based on model configuration."""

    def __init__(self, config: Dict[str, Any]):
        self.primary_tf = config["primary_timeframe"]
        self.mtf_strategy = config["mtf_strategy"]
        self.mtf_sources = config.get("mtf_sources", [])
        self.feature_groups = config["feature_groups"]

    def select_features(
        self,
        canonical_1m: pd.DataFrame,
        mtf_data: Dict[str, pd.DataFrame]
    ) -> np.ndarray:
        """
        Select features based on model config.

        Args:
            canonical_1m: 1-min canonical OHLCV
            mtf_data: Dict of {timeframe: DataFrame} for all 9 TFs

        Returns:
            Feature array (2D, 3D, or 4D based on model type)
        """
        # Resample to primary timeframe
        primary_df = self._resample_to_primary(canonical_1m)

        # Select feature groups
        features = []
        if "indicators" in self.feature_groups:
            features.append(self._compute_indicators(primary_df))
        if "wavelets" in self.feature_groups:
            features.append(self._compute_wavelets(primary_df))
        if "microstructure" in self.feature_groups:
            features.append(self._compute_microstructure(primary_df))
        if "price_volume" in self.feature_groups:
            features.append(self._extract_price_volume(primary_df))
        if "time" in self.feature_groups:
            features.append(self._extract_time_features(primary_df))

        # Add MTF features if strategy requires
        if self.mtf_strategy == "mtf_indicators":
            mtf_features = self._compute_mtf_indicators(mtf_data, self.mtf_sources)
            features.append(mtf_features)

        # Concatenate all features
        feature_matrix = pd.concat(features, axis=1)

        return feature_matrix.values

    def _resample_to_primary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Resample canonical 1-min to primary timeframe."""
        if self.primary_tf == "1min":
            return df

        # Resample OHLCV
        rule = self.primary_tf.replace("min", "T")
        resampled = df.resample(rule).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum"
        })
        return resampled.dropna()

    def _compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute base indicators on primary timeframe."""
        # RSI, MACD, ATR, Bollinger, ADX, etc.
        # (~60 features)
        pass

    def _compute_wavelets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute wavelet decomposition on primary timeframe."""
        # Db4, Haar (3 levels each)
        # (~24 features)
        pass

    def _compute_microstructure(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute microstructure proxies on primary timeframe."""
        # Roll spread, Kyle's lambda, Amihud, etc.
        # (~10 features)
        pass

    def _extract_price_volume(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract price/volume raw features."""
        # Returns, log volume, OHLC ratios
        # (~8 features)
        pass

    def _extract_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract time features."""
        # Hour, day_of_week
        # (~2 features)
        pass

    def _compute_mtf_indicators(
        self,
        mtf_data: Dict[str, pd.DataFrame],
        sources: List[str]
    ) -> pd.DataFrame:
        """Compute MTF indicators from specified sources."""
        # For each source timeframe, compute indicators
        # Align to primary TF via forward-fill + shift(1)
        # (~50 features for 3 sources × ~17 indicators each)
        pass
```

---

### Adapter Integration

**Location:** `src/phase5/adapters/tabular_adapter.py`

```python
from src.phase5.adapters.feature_selector import FeatureSelector

class TabularAdapter:
    """Adapter for tabular models (2D input)."""

    def __init__(self, model_config: Dict[str, Any]):
        self.feature_selector = FeatureSelector(model_config)

    def prepare_data(
        self,
        canonical_1m: pd.DataFrame,
        mtf_data: Dict[str, pd.DataFrame]
    ) -> TimeSeriesDataContainer:
        """Prepare 2D tabular data."""

        # Select features based on model config
        X = self.feature_selector.select_features(canonical_1m, mtf_data)

        # Extract labels
        y = canonical_1m.loc[X.index, "label"].values

        # Split into train/val/test
        X_train, X_val, X_test = self._split_data(X)
        y_train, y_val, y_test = self._split_data(y)

        return TimeSeriesDataContainer(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test
        )
```

---

## Key Constraints

- ✅ All models use same underlying 1-min canonical OHLCV
- ✅ All models use same timestamps (aligned after resampling)
- ✅ All models use same target labels
- ✅ All models use same train/val/test splits
- ✅ Each model gets DIFFERENT features tailored to its needs
- ✅ Feature selection is deterministic and reproducible

**Violations of these constraints will break reproducibility and fairness.**

---

## Comparison: Same Data vs Same Features

### What's the Same (Reproducibility)

| Property | Status | Why |
|----------|--------|-----|
| **Source data** | ✅ Same | All models read `data/raw/MES_1m.parquet` |
| **Timestamps** | ✅ Same | All aligned to canonical 1-min index |
| **Labels** | ✅ Same | Triple-barrier labels computed once |
| **Splits** | ✅ Same | Train/val/test splits with purge/embargo |
| **Leakage prevention** | ✅ Same | shift(1), purge, embargo applied uniformly |

### What's Different (Diversity)

| Property | Status | Why |
|----------|--------|-----|
| **Primary timeframe** | ❌ Different | Each model has optimal TF (5min vs 15min vs 1h) |
| **Feature engineering** | ❌ Different | Tabular gets indicators, transformers get raw OHLCV |
| **MTF strategy** | ❌ Different | MTF indicators vs single-TF vs MTF ingestion |
| **Input shape** | ❌ Different | 2D vs 3D vs 4D based on model family |
| **Feature count** | ❌ Different | ~150-200 features based on groups enabled |

---

## Roadmap

### Phase 1: Single-TF Baselines (Completed)
- ✅ All models use same primary timeframe (5min)
- ✅ No MTF features
- ✅ ~150 base features for all

### Phase 2: MTF Indicators for Tabular (In Progress)
- ⚠️ Tabular models get MTF indicators (5 of 9 TFs implemented)
- ⚠️ Sequence models still single-TF
- ⚠️ Advanced models not implemented

### Phase 3: Per-Model Primary TF (Planned)
- ❌ Configurable primary timeframe per model
- ❌ CatBoost → 15min, TCN → 5min, Ridge → 1h
- ❌ Feature selector respects model config

### Phase 4: MTF Ingestion for Transformers (Planned)
- ❌ PatchTST, iTransformer, TFT get multi-stream raw OHLCV
- ❌ 4D adapter implemented
- ❌ All 9 timeframes upscaled

### Phase 5: Ensemble Diversity Validation (Planned)
- ❌ Validate error correlation reduced vs homogeneous
- ❌ Measure feature overlap across base models
- ❌ Confirm meta-learner performance gains

---

## References

- **MTF Strategies:** `docs/roadmaps/MTF_IMPLEMENTATION_ROADMAP.md`
- **Advanced Models:** `docs/roadmaps/ADVANCED_MODELS_ROADMAP.md`
- **Heterogeneous Ensembles:** `.serena/knowledge/heterogeneous_ensemble_architecture.md`
- **Unified Pipeline:** `.serena/knowledge/unified_pipeline_architecture.md`

---

**Last Updated:** 2026-01-01
