# Implementation Tasks: Current → Intended Architecture

**Date:** 2026-01-01
**Purpose:** Consolidated implementation roadmap for model-specific MTF pipelines
**Scope:** Phase 1 pipeline refactoring + model integration
**Effort:** 12-16 days (code) + 2-4 weeks (testing/validation) = **6-8 weeks total**

---

## Executive Summary

**Goal:** Transform the current universal indicator pipeline into a model-specific multi-timeframe (MTF) system supporting 3 strategies.

**Current State:** All models receive ~180 indicator-derived features from 5 timeframes (universal pipeline).

**Target State:** Model-specific pipelines with 3 MTF strategies:
1. **Strategy 1 (Single-TF):** Train on one timeframe, no MTF - All models
2. **Strategy 2 (MTF Indicators):** Indicator features from 9 timeframes - Tabular models
3. **Strategy 3 (MTF Ingestion):** Raw OHLCV from 9 timeframes - Sequence models

**Total Effort:**
- Code implementation: 12-16 days
- Testing & validation: 2-4 weeks
- **Total: 6-8 weeks (1 engineer) | 4-5 weeks (2 engineers)**

---

## Table of Contents

1. [Component Readiness Assessment](#component-readiness-assessment)
2. [Implementation Phases](#implementation-phases)
3. [File-Level Changes](#file-level-changes)
4. [Testing & Validation](#testing--validation)
5. [Success Criteria](#success-criteria)
6. [Risk Assessment](#risk-assessment)
7. [Parallel Work Opportunities](#parallel-work-opportunities)

---

## Component Readiness Assessment

### Overview: Code is 80% Ready

| Component | Readiness | Effort to Modify | Notes |
|-----------|-----------|------------------|-------|
| **MTF Infrastructure** | **HIGH** | 1-2 days | `MTFMode` enum exists, just needs model routing |
| **Pipeline Config** | **HIGH** | 1 day | Add `mtf_strategy` and `training_timeframe` fields |
| **Model Registry** | **HIGH** | 1 day | Add `supports_mtf_*` metadata |
| **Data Preparation** | **MEDIUM** | 3-4 days | Simple switch logic needed, no major rewrite |
| **Feature Engineering** | **MEDIUM** | 2-3 days | Already modular, needs group definitions |
| **TimeSeriesDataContainer** | **MEDIUM** | 4-5 days | Add `get_multi_resolution_bars()` method |

**Overall Assessment:** The codebase is well-structured and READY for model-specific pipelines with MINOR refactoring.

---

### 1. MTF Infrastructure (HIGH Readiness)

**Location:** `src/phase1/stages/mtf/`

**What Exists:**
```python
# constants.py - ALREADY EXISTS
class MTFMode(str, Enum):
    BARS = 'bars'           # Only OHLCV at higher TFs
    INDICATORS = 'indicators'  # Only indicators at higher TFs
    BOTH = 'both'           # Both OHLCV + indicators (CURRENT DEFAULT)
```

**Current Usage:**
```python
# generator.py, line 79
self.mode = mode  # But always defaults to MTFMode.BOTH
```

**Readiness:** ✅ READY FOR MODEL-SPECIFIC ROUTING
- MTFMode infrastructure already exists
- Generator accepts mode parameter
- Just need to wire mode selection to model family

**Changes Required:**
1. Add model family → MTF mode mapping (5 lines of code)
2. Pass mode from config through pipeline stages
3. Add CLI flag `--mtf-mode` or auto-detect from `--model`

**Estimated Effort:** 1-2 days

---

### 2. Pipeline Configuration (HIGH Readiness)

**Location:** `src/phase1/pipeline_config.py`

**Current Config Fields:**
```python
@dataclass
class PipelineConfig:
    symbol: str
    horizons: list[int]
    purge_bars: int
    embargo_bars: int
    # NO mtf_strategy field
    # NO training_timeframe field
```

**Readiness:** ✅ EASY TO EXTEND
- Uses dataclass pattern (clean extension)
- YAML configs already support model-specific settings
- No breaking changes required

**Changes Required:**
1. Add `mtf_strategy: str` field (single_tf, mtf_indicators, mtf_ingestion)
2. Add `training_timeframe: str` field (5min, 15min, etc.)
3. Add validation for strategy/model compatibility
4. Update CLI to accept new parameters

**Estimated Effort:** 1 day

---

### 3. Model Registry (HIGH Readiness)

**Location:** `src/models/registry.py`

**Current Registration:**
```python
@register(name="xgboost", family="boosting")
class XGBoostModel(BaseModel):
    ...
```

**Readiness:** ✅ READY FOR METADATA EXTENSION
- Clean decorator pattern
- Easy to add metadata fields
- No breaking changes to existing models

**Changes Required:**
1. Add strategy support metadata to registration
2. Add recommended MTF strategy per model
3. Add validation helper

**Estimated Effort:** 1 day

---

### 4. Data Preparation (MEDIUM Readiness)

**Location:** `src/models/data_preparation.py`

**Current Logic:**
```python
def prepare_training_data(container, requires_sequences, sequence_length=60):
    if requires_sequences:
        train_dataset = container.get_pytorch_sequences("train", seq_len=sequence_length)
        X_train, y_train, w_train = dataset_to_arrays(train_dataset)
    else:
        X_train, y_train, w_train = container.get_sklearn_arrays("train")
    return X_train, y_train, w_train, X_val, y_val
```

**Observation:** Already has shape-based routing (2D vs 3D). Just needs FEATURE-based routing.

**Readiness:** ✅ READY FOR ENHANCEMENT
- Clear switch logic already exists
- Can add model family parameter easily
- Container interface supports multiple data retrieval methods

**Changes Required:**
1. Add `model_family` parameter to `prepare_training_data()`
2. Add feature selection logic based on strategy
3. Add multi-resolution support when Strategy 3 is implemented

**Estimated Effort:** 3-4 days

---

### 5. Feature Engineering (MEDIUM Readiness)

**Location:** `src/phase1/stages/features/`

**Current Behavior:**
- Generates ALL features unconditionally
- No model-aware feature selection
- All ~180 features passed to all models

**Readiness:** ✅ MODULAR AND READY
- Feature functions are well-isolated
- Easy to make feature groups selectable
- Existing wavelet/microstructure functions can be optional

**Changes Required:**
1. Define feature groups (returns, indicators, wavelets, mtf, etc.)
2. Add group selection parameter to engineer_features()
3. Create model-family → feature-groups mapping

**Estimated Effort:** 2-3 days

---

### 6. TimeSeriesDataContainer (MEDIUM Readiness)

**Location:** `src/phase1/stages/datasets/container.py`

**Current Methods:**
- `get_sklearn_arrays(split)` - Returns 2D arrays for tabular models
- `get_pytorch_sequences(split, seq_len)` - Returns 3D sequences for neural models
- `get_neuralforecast_df(split)` - Returns DataFrame for NeuralForecast

**Missing Methods (for Strategy 3):**
- `get_multi_resolution_bars(split, timeframes)` - Returns dict of raw OHLCV tensors

**Readiness:** ✅ ARCHITECTURE IS READY
- Container pattern is clean and extensible
- Already has multiple data format methods
- No major refactoring needed to add new method

**Changes Required:**
1. Add `get_multi_resolution_bars()` method (new functionality)
2. Store raw OHLCV at multiple timeframes during pipeline
3. Add tensor utilities for stacking/concatenating

**Estimated Effort:** 4-5 days (this is the biggest change)

---

## Implementation Phases

### Overview: 3 Phases Over 6-8 Weeks

```
Phase 1: Quick Wins (Week 1)
    ├── Enable model-aware MTF routing (1 day)
    ├── Create feature group definitions (2 days)
    └── Integrate feature selection (3 days)

Phase 2: Complete MTF Implementation (Weeks 2-3)
    ├── Add missing timeframes (2 days)
    ├── Implement single-TF baseline (1 day)
    └── Build multi-resolution sequences (5-7 days)

Phase 3: Advanced Features (Week 4+)
    └── Model-specific data prep hooks (3-4 days)
```

---

## Phase 1: Quick Wins (Week 1)

**Goal:** Enable differentiated data preparation for tabular vs. sequence models

### Task 1.1: Enable Model-Aware MTF Routing (1 day)

**Goal:** Tabular models get indicators, sequence models get raw bars

**Files to Modify:**
- `src/phase1/stages/features/run.py`
- `src/pipeline/config.py`

**Changes:**

**File:** `src/phase1/stages/features/run.py`
```python
# Line 64-68: Replace hardcoded mtf_mode
def run_feature_engineering(
    config: 'PipelineConfig',
    manifest: 'ArtifactManifest',
    model_family: str | None = None  # ← ADD parameter
) -> 'StageResult':

    # Determine MTF mode based on model family
    mtf_mode = getattr(config, 'mtf_mode', None)
    if mtf_mode is None and model_family:
        # Auto-detect mode from model family
        if model_family in ["boosting", "classical"]:
            mtf_mode = 'indicators'  # Tabular models
        elif model_family in ["neural", "cnn", "advanced"]:
            mtf_mode = 'bars'  # Sequence models
        else:
            mtf_mode = 'both'  # Ensemble or unknown
    else:
        mtf_mode = mtf_mode or 'both'  # Default
```

**File:** `src/pipeline/config.py`
```python
@dataclass
class PipelineConfig:
    # ... existing fields ...

    # MTF configuration
    mtf_mode: str | None = None  # ← ADD: 'bars', 'indicators', 'both', or None (auto)
    mtf_timeframes: list[str] = field(default_factory=lambda: ['15min', '30min', '1h', '4h', 'daily'])
```

**Testing:**
```bash
# Test tabular model (should get indicators only)
./pipeline run --symbols MES --mtf-mode indicators
python scripts/train_model.py --model xgboost --horizon 20

# Test sequence model (should get bars only)
./pipeline run --symbols MES --mtf-mode bars
python scripts/train_model.py --model lstm --horizon 20 --seq-len 60

# Verify feature counts differ
```

**Expected Outcome:**
- Tabular models: ~143 features (180 - 25 MTF bars)
- Sequence models: ~123 features (180 - 45 MTF indicators)

---

### Task 1.2: Create Feature Group Definitions (2 days)

**Goal:** Organize features into logical groups for model selection

**New File:** `src/phase1/utils/feature_groups.py`

```python
"""
Feature group definitions for model-specific data preparation.

Organizes ~180 total features into logical groups that different
model families can selectively use.
"""
from dataclasses import dataclass
from typing import Literal

FeatureGroup = Literal[
    "base_ohlcv",
    "base_indicators",
    "mtf_bars",
    "mtf_indicators",
    "microstructure",
    "wavelets"
]

@dataclass
class FeatureGroupDefinition:
    """Definition of a feature group."""
    name: FeatureGroup
    description: str
    patterns: list[str]  # Regex patterns to match columns
    count_estimate: int

# Base features (5min timeframe)
BASE_OHLCV = FeatureGroupDefinition(
    name="base_ohlcv",
    description="Base timeframe OHLCV (open, high, low, close, volume)",
    patterns=[r"^(open|high|low|close|volume)$"],
    count_estimate=5
)

BASE_INDICATORS = FeatureGroupDefinition(
    name="base_indicators",
    description="Indicators computed on base timeframe",
    patterns=[
        r"^(return|sma|ema|rsi|macd|stoch|williams|roc|cci|mfi)_",
        r"^(atr|bb_|kc_|hvol|.*_vol)_",
        r"^(volume_|obv|vwap|dollar_volume)",
        r"^(adx|supertrend|hour_|day_|vol_regime|trend_regime)",
        r"^(autocorr|clv|skewness|kurtosis)"
    ],
    count_estimate=64
)

MTF_BARS = FeatureGroupDefinition(
    name="mtf_bars",
    description="OHLCV from higher timeframes (15min, 30min, 1h, 4h, daily)",
    patterns=[r"^(open|high|low|close|volume)_(15m|30m|1h|4h|1d)$"],
    count_estimate=25  # 5 bars × 5 timeframes
)

MTF_INDICATORS = FeatureGroupDefinition(
    name="mtf_indicators",
    description="Indicators computed on higher timeframes",
    patterns=[
        r"^(sma|ema|rsi|atr|bb_position|macd_hist|close_sma20_ratio)_.*_(15m|30m|1h|4h|1d)$"
    ],
    count_estimate=45  # 9 indicators × 5 timeframes
)

MICROSTRUCTURE = FeatureGroupDefinition(
    name="microstructure",
    description="Microstructure proxy features (spread, liquidity, impact)",
    patterns=[r"^micro_"],
    count_estimate=10
)

WAVELETS = FeatureGroupDefinition(
    name="wavelets",
    description="Wavelet decomposition features (multi-scale)",
    patterns=[r"^wavelet_"],
    count_estimate=24
)

# Feature group registry
FEATURE_GROUPS: dict[FeatureGroup, FeatureGroupDefinition] = {
    "base_ohlcv": BASE_OHLCV,
    "base_indicators": BASE_INDICATORS,
    "mtf_bars": MTF_BARS,
    "mtf_indicators": MTF_INDICATORS,
    "microstructure": MICROSTRUCTURE,
    "wavelets": WAVELETS,
}

# Model family recommendations
MODEL_FEATURE_RECOMMENDATIONS = {
    "boosting": ["base_indicators", "mtf_indicators", "microstructure", "wavelets"],
    "classical": ["base_indicators", "mtf_indicators", "microstructure"],
    "neural": ["base_ohlcv", "mtf_bars", "base_indicators"],
    "cnn": ["base_ohlcv", "mtf_bars"],
    "advanced": ["base_ohlcv", "mtf_bars"],  # For multi-resolution models
}

def get_feature_mask(
    feature_columns: list[str],
    include_groups: list[FeatureGroup]
) -> list[bool]:
    """
    Create boolean mask for feature selection.

    Args:
        feature_columns: All available feature names
        include_groups: List of feature groups to include

    Returns:
        Boolean mask: True for features in included groups
    """
    import re

    mask = [False] * len(feature_columns)

    for group_name in include_groups:
        group = FEATURE_GROUPS[group_name]
        for i, col in enumerate(feature_columns):
            if mask[i]:  # Already included
                continue
            for pattern in group.patterns:
                if re.match(pattern, col):
                    mask[i] = True
                    break

    return mask

def filter_features_by_groups(
    feature_columns: list[str],
    include_groups: list[FeatureGroup]
) -> list[str]:
    """Filter feature list to only include specified groups."""
    mask = get_feature_mask(feature_columns, include_groups)
    return [col for col, include in zip(feature_columns, mask) if include]
```

**Testing:**
```python
# Test feature group filtering
from src.phase1.utils.feature_groups import filter_features_by_groups

all_features = [...]  # From TimeSeriesDataContainer
boosting_features = filter_features_by_groups(
    all_features,
    include_groups=["base_indicators", "mtf_indicators", "microstructure", "wavelets"]
)
print(f"Boosting features: {len(boosting_features)}/180")

sequence_features = filter_features_by_groups(
    all_features,
    include_groups=["base_ohlcv", "mtf_bars", "base_indicators"]
)
print(f"Sequence features: {len(sequence_features)}/180")
```

---

### Task 1.3: Integrate Feature Selection into Pipeline (3 days)

**Goal:** Auto-reduce 180 features before training using walk-forward selection

**File:** `src/phase1/stages/datasets/run.py`

```python
# Add after building TimeSeriesDataContainer, before saving
def build_datasets(config, manifest):
    container = TimeSeriesDataContainer.from_parquet_dir(...)

    # Feature selection (optional, controlled by config)
    if config.enable_feature_selection:
        logger.info("Running walk-forward feature selection...")
        from src.cross_validation.feature_selector import WalkForwardFeatureSelector
        from src.cross_validation.purged_kfold import PurgedKFold

        # Get train data
        X_train, y_train, w_train = container.get_sklearn_arrays("train")
        label_end_times = container.get_label_end_times("train")

        # Create CV splits for selection
        cv = PurgedKFold(
            n_splits=config.feature_selection_n_splits,
            purge_bars=config.purge_bars,
            embargo_bars=config.embargo_bars
        )
        cv_splits = list(cv.split(X_train, y_train, label_end_times))

        # Select stable features
        selector = WalkForwardFeatureSelector(
            n_features_to_select=config.n_features_to_select,
            selection_method=config.feature_selection_method,
            min_feature_frequency=config.min_feature_frequency
        )
        result = selector.select_features_walkforward(
            pd.DataFrame(X_train, columns=container.feature_columns),
            pd.Series(y_train),
            cv_splits,
            pd.Series(w_train)
        )

        # Update container with selected features
        logger.info(
            f"Selected {len(result.stable_features)}/{len(container.feature_columns)} "
            f"stable features (min_freq={config.min_feature_frequency})"
        )
        container.config.feature_columns = result.stable_features

        # Save selection metadata
        manifest.add_artifact(
            name="feature_selection_result",
            metadata={
                "n_features_selected": len(result.stable_features),
                "n_features_original": len(X_train[0]),
                "selection_method": config.feature_selection_method,
                "stable_features": result.stable_features,
                "feature_counts": result.feature_counts
            }
        )

    return container
```

**File:** `src/pipeline/config.py`
```python
@dataclass
class PipelineConfig:
    # ... existing fields ...

    # Feature selection
    enable_feature_selection: bool = False
    n_features_to_select: int = 50
    feature_selection_method: str = "mda"  # mda, mdi, hybrid
    feature_selection_n_splits: int = 5
    min_feature_frequency: float = 0.6
```

**Testing:**
```bash
# Run pipeline with feature selection
./pipeline run --symbols MES --enable-feature-selection \
    --n-features 50 --feature-selection-method mda

# Train model (should use 50 selected features, not 180)
python scripts/train_model.py --model xgboost --horizon 20
```

---

## Phase 2: Complete MTF Implementation (Weeks 2-3)

**Goal:** Full 9-timeframe MTF ladder + multi-resolution support

### Task 2.1: Add Missing Timeframes (2 days)

**Goal:** Support full 9-timeframe ladder as documented

**File:** `src/phase1/stages/mtf/constants.py`

```python
# Line 20-39: Add missing timeframes
MTF_TIMEFRAMES = {
    # Base timeframe
    '1min': 1,
    '5min': 5,
    # Short-term MTF
    '10min': 10,
    '15min': 15,
    '20min': 20,   # ← ADD
    '25min': 25,   # ← ADD
    '30min': 30,
    '45min': 45,
    # Hourly
    '60min': 60,
    '1h': 60,
    # Multi-hour
    '4h': 240,
    '240min': 240,
    # Daily
    'daily': 1440,
    '1d': 1440,
    'D': 1440,
}

# Update default to 9-TF ladder (exclude 1min if base is 5min)
DEFAULT_MTF_TIMEFRAMES = [
    '5min', '10min', '15min', '20min', '25min', '30min', '45min', '1h'
]  # 8 timeframes (9 including base 5min context)

# Update PANDAS_FREQ_MAP
PANDAS_FREQ_MAP = {
    # ... existing ...
    '20min': '20min',  # ← ADD
    '25min': '25min',  # ← ADD
}
```

**Impact:** MTF features increase from 70 to ~112 (8 TFs × 14 features/TF)

**Testing:**
```bash
# Test all 9 timeframes
./pipeline run --symbols MES --mtf-timeframes 5min,10min,15min,20min,25min,30min,45min,1h
```

---

### Task 2.2: Implement Single-TF Baseline (1 day)

**Goal:** Support Strategy 1 (no MTF features)

**File:** `src/phase1/stages/features/engineer.py`

```python
# Line 147-151: Make MTF optional
self.enable_mtf = enable_mtf
if not enable_mtf:
    logger.info("MTF features DISABLED - running single-timeframe baseline")
    self.mtf_timeframes = []
```

**File:** `pipeline` CLI or config
```bash
# Add flag
./pipeline run --symbols MES --disable-mtf
```

**Expected Features:**
- Without MTF: ~98 features (180 - 70 MTF)
- Use for baseline comparison: "Does MTF improve performance?"

---

### Task 2.3: Build Multi-Resolution Sequence Dataset (5-7 days)

**Goal:** Enable Strategy 3 (multi-resolution OHLCV for advanced models)

**New File:** `src/phase1/stages/datasets/multiresolution.py`

```python
"""
Multi-resolution sequence dataset for advanced time series models.

Generates aligned multi-resolution OHLCV windows for models like:
- InceptionTime (multi-scale convolutions)
- PatchTST (patched time series)
- iTransformer (inverted attention)
- TFT (temporal fusion)
- N-BEATS (interpretable decomposition)
"""
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class MultiResolutionSequenceDataset(Dataset):
    """
    Dataset that yields multi-resolution OHLCV tensors.

    For each sample at time t, returns dict of windows:
    {
        '5min':  (60, 4) - last 60 bars of 5min OHLC
        '15min': (20, 4) - last 20 bars of 15min OHLC
        '30min': (10, 4) - last 10 bars of 30min OHLC
        '1h':    (5, 4)  - last 5 bars of 1h OHLC
    }

    All windows align to the same prediction point (time t).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        timeframes: list[str],
        bars_per_tf: dict[str, int],
        label_column: str,
        weight_column: str | None = None,
        symbol_column: str | None = "symbol",
    ):
        """
        Args:
            df: DataFrame with OHLCV and MTF bar columns
            timeframes: List of timeframes to include
            bars_per_tf: Window size per timeframe
            label_column: Target label column
            weight_column: Sample weight column
            symbol_column: Symbol identifier for isolation
        """
        self.df = df
        self.timeframes = timeframes
        self.bars_per_tf = bars_per_tf
        self.label_column = label_column
        self.weight_column = weight_column
        self.symbol_column = symbol_column

        # Pre-compute OHLCV column mapping
        self.ohlcv_cols = self._build_ohlcv_mapping()

        # Build valid sequence indices (symbol-isolated)
        self.indices = self._build_indices()

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[dict[str, np.ndarray], int, float]:
        """
        Get multi-resolution sample.

        Returns:
            X_multi: Dict mapping timeframe -> OHLCV array (n_bars, 4)
            y: Label (int)
            weight: Sample weight (float)
        """
        # Implementation...
        pass
```

**Testing:**
```python
# Test multi-resolution dataset
container = TimeSeriesDataContainer.from_parquet_dir(...)
dataset = container.get_multiresolution_sequences("train")

# Check output shape
X_multi, y, w = dataset[0]
assert '5min' in X_multi
assert X_multi['5min'].shape == (60, 4)  # 60 bars, 4 OHLC features
assert X_multi['15min'].shape == (20, 4)
assert X_multi['1h'].shape == (5, 4)
```

---

## Phase 3: Advanced Features (Week 4+)

### Task 3.1: Model-Specific Data Preparation Hooks (3-4 days)

**Goal:** Allow models to customize data preparation

**File:** `src/models/base.py`

```python
class BaseModel(ABC):
    @classmethod
    def prepare_data(
        cls,
        container: TimeSeriesDataContainer,
        split: str,
        **kwargs
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for this model type.

        Override this method to customize feature selection, scaling, or data format.

        Args:
            container: TimeSeriesDataContainer with splits
            split: Split name ("train", "val", "test")
            **kwargs: Model-specific parameters (e.g., seq_len for sequence models)

        Returns:
            Tuple of (X, y, weights)
        """
        # Default implementation
        if cls.requires_sequences:
            seq_len = kwargs.get('seq_len', 60)
            dataset = container.get_pytorch_sequences(split, seq_len=seq_len)
            from src.models.data_preparation import dataset_to_arrays
            return dataset_to_arrays(dataset)
        else:
            return container.get_sklearn_arrays(split)
```

**Example Override:**
```python
# src/models/boosting/xgboost_model.py
class XGBoostModel(BaseModel):
    @classmethod
    def prepare_data(cls, container, split, **kwargs):
        # XGBoost only wants indicator features (no MTF bars)
        from src.phase1.utils.feature_groups import filter_features_by_groups

        X, y, w = container.get_sklearn_arrays(split)

        # Filter to recommended feature groups
        feature_mask = filter_features_by_groups(
            container.feature_columns,
            include_groups=["base_indicators", "mtf_indicators", "microstructure", "wavelets"]
        )

        X_filtered = X[:, feature_mask]
        return X_filtered, y, w
```

---

## File-Level Changes

### Top 5 Most Critical Files

1. **`src/phase1/pipeline_config.py`** (1 day)
   - Add `mtf_strategy`, `training_timeframe`, feature selection config
   - Add validation for strategy/model compatibility

2. **`src/phase1/stages/mtf/generator.py`** (2 days)
   - Add 20min, 25min to timeframe ladder
   - Make base_timeframe configurable (not hardcoded to 5min)
   - Respect `mtf_mode` from config

3. **`src/phase1/stages/datasets/multiresolution.py`** (5-7 days, NEW FILE)
   - Create `MultiResolutionSequenceDataset` class
   - Implement synchronized multi-resolution tensor extraction
   - Add tests for all timeframe combinations

4. **`src/phase1/stages/features/run.py`** (3 days)
   - Add conditional MTF generation based on strategy
   - Wire model_family parameter through pipeline
   - Add feature group filtering

5. **`scripts/train_model.py`** (2 days)
   - Add `--training-timeframe`, `--mtf-strategy` CLI flags
   - Update data loading to use model-specific prep
   - Add validation for strategy/model compatibility

### Additional Important Files

- `src/models/registry.py` (1 day) - Add MTF capability metadata to all models
- `src/inference/pipeline.py` (3 days) - Update inference to support all strategies
- `tests/integration/test_mtf_end_to_end.py` (2 days, NEW) - Comprehensive MTF tests
- `src/phase1/stages/datasets/container.py` (2 days) - Add `get_multi_resolution_bars()`
- `src/phase1/stages/datasets/tensor_utils.py` (1 day, NEW) - Concatenation/stacking utilities

---

## Testing & Validation

### Unit Tests

```python
# tests/test_feature_groups.py
def test_feature_group_filtering():
    from src.phase1.utils.feature_groups import filter_features_by_groups

    all_features = [
        "return_1", "rsi_14", "open_15m", "sma_20_1h", "micro_spread", "wavelet_cA3"
    ]

    indicators = filter_features_by_groups(all_features, ["base_indicators"])
    assert "return_1" in indicators
    assert "rsi_14" in indicators
    assert "open_15m" not in indicators

    mtf_bars = filter_features_by_groups(all_features, ["mtf_bars"])
    assert "open_15m" in mtf_bars
    assert "return_1" not in mtf_bars

# tests/test_multiresolution.py
def test_multiresolution_dataset():
    from src.phase1.stages.datasets.multiresolution import MultiResolutionSequenceDataset

    # Create synthetic data
    df = create_synthetic_ohlcv_with_mtf(n_bars=1000)

    dataset = MultiResolutionSequenceDataset(
        df=df,
        timeframes=['5min', '15min', '30min'],
        bars_per_tf={'5min': 60, '15min': 20, '30min': 10},
        label_column='label_h20'
    )

    X_multi, y, w = dataset[0]
    assert set(X_multi.keys()) == {'5min', '15min', '30min'}
    assert X_multi['5min'].shape == (60, 4)
    assert X_multi['15min'].shape == (20, 4)
```

### Integration Tests

```bash
# Test full pipeline with new features
./pipeline run --symbols MES --mtf-mode indicators --enable-feature-selection

# Train tabular model (should use filtered features)
python scripts/train_model.py --model xgboost --horizon 20

# Train sequence model (should use multi-resolution data)
python scripts/train_model.py --model lstm --horizon 20 --seq-len 60 --use-multiresolution
```

### Validation Checklist

- [ ] MTF mode routing works (tabular → indicators, sequence → bars)
- [ ] Feature groups correctly filter features (test all 6 groups)
- [ ] Feature selection integrates into pipeline (selects stable features)
- [ ] 9-timeframe ladder generates expected feature count (~112 MTF features)
- [ ] Single-TF baseline runs without MTF features
- [ ] Multi-resolution dataset yields correct tensor shapes
- [ ] Model-specific prep hooks allow customization
- [ ] All tests pass (unit + integration)
- [ ] Documentation updated to reflect changes

---

## Success Criteria

### Week 1 (Quick Wins)

- ✅ Tabular models use ~143 features (no MTF bars)
- ✅ Sequence models use ~123 features (no MTF indicators)
- ✅ Feature groups defined and tested
- ✅ Feature selection reduces features to top 50 (configurable)

### Week 2-3 (MTF Completion)

- ✅ All 9 timeframes supported (5min-1h ladder)
- ✅ Single-TF baseline mode available
- ✅ Multi-resolution dataset yields correct tensor shapes
- ✅ Advanced models can consume multi-resolution inputs

### Week 4+ (Advanced)

- ✅ Models can override `prepare_data()` for custom logic
- ✅ Feature importance tracking per model
- ✅ Dynamic feature filtering in container

### Final Deliverable

After implementing all phases:

- [ ] All models receive optimal data for their architecture
- [ ] Tabular models receive ~180 indicator features (Strategy 2)
- [ ] Sequence models can receive raw OHLCV bars (Strategy 3)
- [ ] All models support single-TF baseline (Strategy 1)
- [ ] Config validates model-strategy compatibility
- [ ] Existing pipelines still work (backward compatible)
- [ ] Performance benchmarks show improvement for sequence models with Strategy 3
- [ ] Full test coverage (unit + integration)
- [ ] Documentation complete

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Breaking existing pipelines | Low | High | Add `mtf_strategy='legacy'` default, maintain backward compatibility |
| Performance regression | Low | Medium | Benchmark before/after, ensure default behavior unchanged |
| Test coverage gaps | Medium | Medium | Add strategy-specific tests, maintain >80% coverage |
| Feature count mismatch | Low | Low | Log feature counts at pipeline stages, add validation |
| Multi-resolution tensor shape errors | Medium | Medium | Comprehensive shape testing, clear error messages |
| Model convergence issues with Strategy 3 | Medium | Medium | Add model-specific hyperparameter tuning, early stopping |

---

## Parallel Work Opportunities

### With 2 Engineers: 4-5 Weeks Total

```
Phase 1 (Config + Registry)
    ↓
Phase 2 (MTF Mode)    Phase 3 (Feature Groups)
    ↓                      ↓
    └──────→ Phase 4 (Data Prep) ←──────┘
                  ↓
            Phase 5 (Container)
```

**Engineer A:**
- Phase 1: Config + registry (1 week)
- Phase 2: MTF mode routing (1 week)
- Phase 4: Data preparation (1 week)
- Testing & validation (1-2 weeks)

**Engineer B:**
- Phase 3: Feature groups (1 week)
- Phase 5: Multi-resolution container (2 weeks)
- Testing & validation (1-2 weeks)

**Timeline:** 4-5 weeks with 2 engineers (vs 6-8 weeks solo)

---

## Breaking Changes

**None Expected:** All changes are additive or behind feature flags.

**Migration Path:**
1. Existing models continue to work (default `mtf_strategy='mtf_indicators'`)
2. New models can opt into model-specific modes
3. Old pipelines compatible with new code (backward compatible)

---

## Documentation Updates Required

1. **CLAUDE.md:**
   - Update MTF timeframes: 5 → 9 (when implemented)
   - Add feature group concept
   - Document model-specific data prep
   - Add architecture warnings (current vs intended)

2. **docs/phases/PHASE_1.md:**
   - Clarify Strategy 1/2/3 implementation status
   - Add feature count breakdown by group
   - Update MTF mode usage

3. **docs/CURRENT_VS_INTENDED_ARCHITECTURE.md:**
   - Mark Strategy 2 as "Complete" when 9-TF done
   - Mark Strategy 3 as "Complete" when multi-resolution done
   - Update feature counts

4. **docs/guides/MTF_STRATEGY_GUIDE.md:** (NEW)
   - User guide for choosing strategies
   - Model-strategy compatibility matrix
   - Configuration examples

5. **README.md:**
   - Add quick start for model-specific modes
   - Document `--mtf-mode`, `--enable-feature-selection` flags
   - Update architecture diagram

---

## References

- **MTF Roadmap:** `docs/roadmaps/MTF_IMPLEMENTATION_ROADMAP.md` (6-8 week plan)
- **Current Limitations:** `docs/CURRENT_LIMITATIONS.md`
- **Architecture Analysis:** `docs/CURRENT_VS_INTENDED_ARCHITECTURE.md`
- **Strategy Guide:** `docs/guides/MTF_STRATEGY_GUIDE.md` (user-facing)
- **Phase 1 Details:** `docs/phases/PHASE_1.md`

---

**Conclusion:** The codebase is well-architected and ready for model-specific pipeline implementation. The main work is adding new functionality (Strategy 3 multi-resolution), not refactoring existing code. Existing infrastructure (MTFMode, container pattern, registry) can be leveraged with minimal changes. Total effort: 6-8 weeks (1 engineer) or 4-5 weeks (2 engineers).
