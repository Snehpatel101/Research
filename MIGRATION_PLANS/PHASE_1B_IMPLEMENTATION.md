# PHASE 1B: LABELING & OPTUNA OPTIMIZATION - Implementation Plan

**Status:** ✅ COMPLETE (90%)
**Last Updated:** 2026-01-18
**Dependencies:** PHASE_0 (Foundation), PHASE_1 (Features)

---

## Executive Summary

PHASE_1B handles all label generation and Optuna-based optimization for features, labels, and hyperparameters. This phase provides a 4-stage optimization pipeline that systematically optimizes the ML pipeline.

---

## Current State Analysis

### Package Structure

```
src/labeling/
├── __init__.py              ✅ Complete
├── triple_barrier.py        ✅ Complete - TripleBarrierLabeler + config
└── optimization.py          ✅ Complete - LabelOptimizer with Optuna

src/optimization/
├── __init__.py              ✅ Complete
├── hyperparameters.py       ✅ Complete - 23 model search spaces
├── labels.py                ✅ Complete - LabelOptimizer wrapper
├── features.py              ✅ Complete - FeatureOptimizer wrapper
└── pipeline.py              ✅ Complete - UnifiedOptimizationPipeline

src/features/
├── selection.py             ✅ Complete - FeatureSelector (3 strategies)
└── pruning.py               ✅ Complete - FeaturePruner (3 strategies)
```

---

## Implemented Components

### 1. Triple-Barrier Labeling (`src/labeling/triple_barrier.py`)

```python
# Key exports:
TripleBarrierConfig   # Dataclass with barrier parameters
TripleBarrierLabeler  # Label generator

# Config parameters:
- upper_mult: float = 2.0        # ATR multiplier for profit target
- lower_mult: float = 2.0        # ATR multiplier for stop loss
- horizon: int = 20              # Max holding period (bars)
- atr_period: int = 14           # ATR calculation window
- use_adaptive_barriers: bool    # Volatility scaling
- vol_lookback: int = 60         # Volatility window
- target_*_pct: float            # Target class distribution

# Label classes:
+1 = Long  (upper barrier hit first)
-1 = Short (lower barrier hit first)
 0 = Neutral (timeout - neither hit)
```

### 2. Label Optimization (`src/labeling/optimization.py`)

```python
# Key exports:
LabelOptimizationResult  # Optuna optimization result
LabelOptimizer           # Optuna-based label parameter search

# Optimization objectives (weighted):
- Balance score (60%): Class distribution vs target
- Predictability score (30%): Quick RF validation
- Sample penalty (10%): Minimum samples per class

# Search space:
- upper_mult: [0.5, 4.0]
- lower_mult: [0.5, 4.0]
- horizon: [5, 60]
- atr_period: [7, 28]
```

### 3. Feature Selection (`src/features/selection.py`)

```python
# Key exports:
FeatureSelectionResult  # Selection result with importance
FeatureSelector         # Optuna binary feature selection

# Selection strategies:
1. Binary: Include/exclude each feature (combinatorial)
2. Family: Select entire feature families
3. Importance: Guided by pre-computed importance scores

# Result tracking:
- selected_features: List[str]
- feature_importance: Dict[str, float] (selection frequency)
- best_score: float (CV score)
- improvement: float (vs baseline)
```

### 4. Feature Pruning (`src/features/pruning.py`)

```python
# Key exports:
FeaturePruningResult  # Pruning result with metrics
FeaturePruner         # Multiple pruning strategies

# Pruning strategies:
1. Importance-based: Remove lowest importance features (Optuna)
2. Correlation-based: Remove highly correlated pairs (fast)
3. Null importance: Permutation-based significance test

# Optimization with MedianPruner for early stopping
```

### 5. Hyperparameter Optimization (`src/optimization/hyperparameters.py`)

```python
# Key exports:
HyperparameterResult     # Optimization result
HyperparameterOptimizer  # Optuna hyperparameter search
HYPERPARAMETER_SPACES    # Search spaces for 23 models

# Model coverage:
- Boosting (3): xgboost, lightgbm, catboost
- Classical (3): random_forest, logistic, svm
- Neural RNN (3): lstm, gru, tcn
- Transformer (4): transformer, patchtst, itransformer, tft
- Other Neural (3): nbeats, inceptiontime, resnet1d
- Meta-learners (4): ridge_meta, mlp_meta, xgboost_meta, calibrated_meta
- Ensemble (3): voting, stacking, blending

# Features:
- TPESampler for efficient search
- HyperbandPruner for neural models
- Per-fold reporting for early stopping
```

### 6. Unified Pipeline (`src/optimization/pipeline.py`)

```python
# Key exports:
FullOptimizationResult   # Complete results
OptimizationPipeline     # 4-stage orchestrator

# Pipeline stages:
Stage 1: Label Optimization (100 trials)
    ↓ Best TripleBarrierConfig
Stage 2: Feature Selection (100 trials)
    ↓ Selected feature subset
Stage 3: Feature Pruning (50 trials)
    ↓ Final pruned features
Stage 4: Hyperparameter Optimization (100 trials × N models)
    ↓ Best params per model

# Total trials: ~350-450+ depending on model count
```

---

## Optimization Trial Budget

| Stage | Trials | Purpose |
|-------|--------|---------|
| Label Optimization | 100 | Find balanced barrier params |
| Feature Selection | 100 | Binary include/exclude |
| Feature Pruning | 50 | Importance-based removal |
| Hyperparameters | 100 × N | Per-model search |
| **Total** | **250 + 100N** | Full pipeline |

**Example with 5 models:** 250 + 500 = 750 trials

---

## Remaining Tasks

### Task 1B.1: Verify Label Quality Metrics ⚠️

**Gap:** Need validation that optimized labels are actually better for downstream training.

**Action Items:**
- [ ] Add post-optimization validation comparing random vs optimized labels
- [ ] Track label quality metrics across optimization history
- [ ] Add label distribution visualization

### Task 1B.2: Add Parallelization Support ⚠️

**Gap:** Currently single-threaded for reproducibility.

**Action Items:**
- [ ] Add `n_jobs` parameter to OptimizationPipeline
- [ ] Use joblib for parallel trials where reproducibility isn't critical
- [ ] Add progress tracking for parallel execution

### Task 1B.3: Cache Optimization Results ⚠️

**Gap:** No caching of intermediate results.

**Action Items:**
- [ ] Add result caching to avoid re-running expensive optimizations
- [ ] Implement hash-based cache keys from config
- [ ] Add cache invalidation when data changes

---

## Integration Points

| Downstream Phase | Consumes |
|------------------|----------|
| PHASE_2 | `TripleBarrierConfig` for label generation |
| PHASE_3 | `final_hyperparams` for model training |
| PHASE_3 | `final_features` for feature selection |
| PHASE_5 | Optimization metadata for bundle |

---

## Usage Examples

### Example 1: Label Optimization Only
```python
from src.labeling import LabelOptimizer, TripleBarrierLabeler

# Optimize label parameters
optimizer = LabelOptimizer(n_trials=100)
result = optimizer.optimize(ohlcv_df, feature_df)

print(f"Best score: {result.best_score:.4f}")
print(f"Best config: {result.best_config}")
print(f"Distribution: {result.class_distribution}")

# Generate labels with best config
labeler = TripleBarrierLabeler(result.best_config)
labels = labeler.create_labels(ohlcv_df)
```

### Example 2: Full Optimization Pipeline
```python
from src.optimization import OptimizationPipeline

pipeline = OptimizationPipeline(
    label_trials=100,
    feature_trials=100,
    pruning_trials=50,
    hyperparam_trials=100,
)

result = pipeline.run_full_optimization(
    ohlcv_df=ohlcv_df,
    feature_df=feature_df,
    models=["xgboost", "lightgbm", "lstm"],
    model_factories=model_factories,
)

print(f"Total trials: {result.total_trials}")
print(f"Final features: {len(result.final_features)}")
print(f"Time: {result.optimization_time_seconds:.1f}s")

# Use optimized parameters
for model_name, params in result.final_hyperparams.items():
    print(f"{model_name}: {params}")
```

### Example 3: Feature Selection with Custom Model
```python
from src.features import FeatureSelector
from xgboost import XGBClassifier

selector = FeatureSelector(
    n_trials=100,
    selection_strategy="binary",
    min_features=20,
    max_features=100,
)

result = selector.select_features(
    X=features.values,
    y=labels,
    feature_names=features.columns.tolist(),
    model_fn=lambda: XGBClassifier(n_estimators=100),
)

print(f"Selected {result.n_selected} from {result.n_total}")
print(f"Top features: {sorted(result.feature_importance.items(), key=lambda x: -x[1])[:10]}")
```

---

## Configuration in PipelineConfig

```python
# Already integrated in src/core/config.py:
@dataclass
class PipelineConfig:
    # Label optimization
    optimize_labels: bool = True
    label_optimization_trials: int = 100
    target_class_distribution: Optional[Dict] = None

    # Feature optimization
    optimize_features: bool = True
    feature_selection_trials: int = 100
    feature_pruning_trials: int = 50
    min_features: int = 20

    # Hyperparameter optimization
    optimize_hyperparams: bool = True
    hyperparam_trials: int = 100

    # Optuna settings
    optuna_random_state: int = 42
```

---

## Sign-off Criteria

- [x] TripleBarrierLabeler with ATR-based barriers
- [x] LabelOptimizer with balance + predictability scoring
- [x] FeatureSelector with 3 strategies
- [x] FeaturePruner with 3 strategies
- [x] HyperparameterOptimizer with 23 model spaces
- [x] OptimizationPipeline with 4 stages
- [ ] Post-optimization label validation
- [ ] Parallelization support
- [ ] Result caching

**PHASE_1B Status: READY FOR PHASE_2**
