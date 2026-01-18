# PHASE 1: UNIFIED FEATURES - Implementation Plan

**Status:** ✅ COMPLETE (90%)
**Last Updated:** 2026-01-18
**Dependencies:** PHASE_0 (Foundation)

---

## Executive Summary

PHASE_1 creates a unified feature system with 162 base features across 12 families, per-model feature strategies, and Optuna-based optimization. This phase is **substantially complete**.

---

## Current State Analysis

### Package Structure

```
src/features/
├── __init__.py              ✅ Complete - All exports defined
├── registry.py              ✅ Complete - FeatureRegistry singleton
├── strategies.py            ✅ Complete - 22 model strategies
├── strategy_manager.py      ✅ Complete - Feature resolution
├── optimization.py          ✅ Complete - Optuna optimization
├── selection.py             ✅ Complete - Feature selection (PHASE_1B)
├── pruning.py               ✅ Complete - Feature pruning (PHASE_1B)
└── compute/
    ├── __init__.py          ✅ Complete - Unified FEATURE_COMPUTE_MAP
    ├── raw.py               ✅ Complete - 5 features
    ├── momentum.py          ✅ Complete - 23 features
    ├── moving_average.py    ✅ Complete - 16 features
    ├── volatility.py        ✅ Complete - 25 features
    ├── volume.py            ✅ Complete - 15 features
    ├── trend.py             ✅ Complete - 6 features
    ├── price.py             ✅ Complete - 12 features
    ├── microstructure.py    ✅ Complete - 15 features
    ├── entropy.py           ✅ Complete - 12 features
    ├── wavelets.py          ✅ Complete - 15 features
    ├── temporal.py          ✅ Complete - 9 features
    └── regime.py            ✅ Complete - 9 features
```

### Feature Count Verification

| Family | Expected | Implemented | Status |
|--------|----------|-------------|--------|
| raw | 5 | 5 | ✅ |
| momentum | 23 | 23 | ✅ |
| moving_average | 16 | 16 | ✅ |
| volatility | 25 | 25 | ✅ |
| volume | 15 | 15 | ✅ |
| trend | 6 | 6 | ✅ |
| price | 12 | 12 | ✅ |
| microstructure | 15 | 15 | ✅ |
| entropy | 12 | 12 | ✅ |
| wavelets | 15 | 15 | ✅ |
| temporal | 9 | 9 | ✅ |
| regime | 9 | 9 | ✅ |
| **TOTAL** | **162** | **162** | ✅ |

---

## Implemented Components

### 1. FeatureRegistry (`registry.py`)
```python
# Key exports:
FeatureDefinition      # Dataclass for feature metadata
FEATURE_REGISTRY       # Singleton dict of all features
get_features_by_families(families: List[str]) -> List[str]
get_features_by_model(model_name: str) -> List[str]
get_feature_families() -> List[str]
```

### 2. Feature Computation (`compute/__init__.py`)
```python
# Main computation functions:
compute_all_features(df) -> DataFrame       # All 162 features
compute_features_by_family(df, families)    # Specific families
compute_features_by_names(df, feature_names)  # Specific features
compute_single_feature(df, feature_name)    # Single feature

# Feature maps:
FEATURE_COMPUTE_MAP    # Dict[str, Callable] - All 162 features
FAMILY_FEATURE_MAPS    # Dict[family, Dict[feature, Callable]]
FEATURE_TO_FAMILY      # Dict[feature, family]
```

### 3. Model Feature Strategies (`strategies.py`)
```python
# Key exports:
ModelFeatureStrategy   # Dataclass with baseline config
MODEL_FEATURE_STRATEGIES  # Dict for all 22 models
get_strategy_for_model(model_name) -> ModelFeatureStrategy
get_baseline_features(model_name) -> List[str]
```

**22 Model Strategies Defined:**
- Boosting (3): xgboost, lightgbm, catboost
- Classical (3): random_forest, logistic, svm
- Neural RNN (3): lstm, gru, tcn
- Transformer (4): transformer, patchtst, itransformer, tft
- Other Neural (3): nbeats, inceptiontime, resnet1d
- Meta-learners (4): ridge_meta, mlp_meta, xgboost_meta, calibrated_meta
- Ensemble (3): voting, stacking, blending

### 4. Feature Optimization (`optimization.py`)
```python
# Key exports:
OptimizationResult     # Dataclass with optimization results
FeatureOptimizer       # Optuna-based feature pruning
optimize_features_for_model(X, y, model_name) -> OptimizationResult
suggest_features(model_name) -> List[str]
```

### 5. Feature Selection (`selection.py`) - PHASE_1B
```python
# Key exports:
FeatureSelectionResult   # Dataclass with selection results
FeatureSelector          # Optuna binary selection
select_features(X, y, feature_names, model_fn) -> FeatureSelectionResult
```

**Selection Strategies:**
- Binary: Include/exclude each feature
- Family: Select entire families
- Importance: Use pre-computed importance scores

### 6. Feature Pruning (`pruning.py`) - PHASE_1B
```python
# Key exports:
FeaturePruningResult    # Dataclass with pruning results
FeaturePruner           # Multiple pruning strategies
prune_features(X, y, feature_names, model_fn) -> FeaturePruningResult
prune_correlated_features(X, feature_names) -> List[str]
```

**Pruning Strategies:**
- Importance-based: Optuna threshold optimization
- Correlation-based: Remove highly correlated pairs
- Null importance: Permutation-based significance testing

---

## Remaining Tasks

### Task 1.1: Add MTF Feature Computation ⚠️

**Gap:** MTF (Multi-Timeframe) features not yet integrated into compute pipeline.

**Required:**
```python
# In src/features/compute/mtf.py (NEW FILE)
def compute_mtf_features(
    df_1min: pd.DataFrame,
    higher_tf_dfs: Dict[str, pd.DataFrame],  # {"5min": df_5min, "15min": df_15min, ...}
    base_features: List[str] = ["rsi_14", "atr_14", "macd_line"],
) -> pd.DataFrame:
    """
    Compute MTF features with shift(1) anti-lookahead.

    Returns ~30 features per higher TF (8 TFs = ~240 MTF features).
    """
    pass
```

**Action Items:**
- [ ] Create `src/features/compute/mtf.py`
- [ ] Implement resampling with proper alignment
- [ ] Apply shift(1) for anti-lookahead
- [ ] Register MTF features in FEATURE_COMPUTE_MAP
- [ ] Update FEATURE_FAMILY_COUNTS

### Task 1.2: Validate Wavelet Dependencies

**Gap:** Wavelets require `pywt` which may not be installed.

**Verification:**
```python
# Already in wavelets.py:
PYWT_AVAILABLE = True/False  # Based on import attempt

# Need to verify graceful fallback when pywt not available
```

**Action Items:**
- [ ] Verify wavelet features return NaN when pywt unavailable
- [ ] Add clear warning message
- [ ] Document optional dependency

### Task 1.3: Verify Feature Computation Performance

**Gap:** No benchmarks for feature computation time.

**Action Items:**
- [ ] Add timing to `compute_all_features()`
- [ ] Identify slow features (entropy, wavelets likely candidates)
- [ ] Consider parallel computation for independent features

---

## Integration Points

| Downstream Phase | Consumes |
|------------------|----------|
| PHASE_1B | `FeatureSelector`, `FeaturePruner`, `compute_all_features` |
| PHASE_2 | `get_strategy_for_model()`, feature column names |
| PHASE_3 | `FEATURE_COMPUTE_MAP` for training data prep |
| PHASE_5 | Feature lineage for PreprocessingGraph |

---

## Usage Examples

### Example 1: Compute All Features
```python
from src.features import compute_all_features

# Load OHLCV data
df = pd.read_parquet("data/mes_1min.parquet")

# Compute all 162 features
features = compute_all_features(df)
print(features.shape)  # (n_rows, 162)
```

### Example 2: Get Features for Model
```python
from src.features import get_strategy_for_model, get_baseline_features

# Get XGBoost strategy
strategy = get_strategy_for_model("xgboost")
print(strategy.baseline_families)  # ['momentum', 'volatility', ...]
print(strategy.min_features)  # 40
print(strategy.max_features)  # 150

# Get actual feature names
features = get_baseline_features("xgboost")
print(len(features))  # ~80 features
```

### Example 3: Feature Selection with Optuna
```python
from src.features import select_features
from sklearn.ensemble import RandomForestClassifier

result = select_features(
    X=features.values,
    y=labels,
    feature_names=features.columns.tolist(),
    model_fn=lambda: RandomForestClassifier(n_estimators=100),
    n_trials=100,
    strategy="binary",
)
print(f"Selected {result.n_selected} from {result.n_total}")
print(f"Improvement: {result.improvement:.2%}")
```

---

## Migration Steps (For External Code)

```python
# OLD - Scattered feature computation
from somewhere.features import compute_rsi, compute_atr
rsi = compute_rsi(df, 14)
atr = compute_atr(df, 14)

# NEW - Unified feature system
from src.features import compute_features_by_names
features = compute_features_by_names(df, ["rsi_14", "atr_14"])
```

---

## Sign-off Criteria

- [x] All 12 feature families implemented
- [x] All 162 base features registered
- [x] All 22 model strategies defined
- [x] FeatureSelector with 3 strategies
- [x] FeaturePruner with 3 strategies
- [ ] MTF features implemented
- [ ] Performance benchmarks added
- [ ] Unit tests for all families

**PHASE_1 Status: READY FOR PHASE_1B (MTF pending)**
