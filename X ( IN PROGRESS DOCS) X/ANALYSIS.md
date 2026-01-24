# ML Factory: Direction & Analysis

**Generated:** 2026-01-23
**Status:** Phases 0-2 Complete | Phase 3 Identified

Comprehensive architecture analysis synthesizing 18 architectural inconsistencies, code quality metrics, data flow gaps, and error handling deficiencies across the ML Factory codebase.

---

## Table of Contents
- [Current State](#current-state)
- [Architecture Overview](#architecture-overview)
- [Data Flows](#data-flows)
- [Decision Trees](#decision-trees)
- [Refactoring Trajectory](#refactoring-trajectory)
- [Cleanup Phases](#cleanup-phases)
- [What NOT to Do](#what-not-to-do)
- [Summary](#summary)

---

## Current State

### Executive Summary

The ML Factory is a config-driven system for building financial time-series ensembles. While the core vision is sound, the implementation has accumulated significant technical debt across four dimensions:

| Category | Critical | High | Medium | Total |
|----------|----------|------|--------|-------|
| Architectural Inconsistencies | 2 | 6 | 3 | 11 |
| Code Quality Flaws | 2 | 3 | 3 | 8 |
| Data Flow Gaps | 3 | 3 | 2 | 8 |
| Error Handling Gaps | 7 | 4 | 3 | 14 |
| **Total** | **14** | **16** | **11** | **41** |

### Codebase Metrics

```
+---------------------------+------------+
| Metric                    | Count      |
+---------------------------+------------+
| Total Python Files        | 449        |
| Total Lines of Code       | ~85,000    |
| Dead Code Functions       | 588        |
| Functions > 50 Lines      | 562        |
| Import Cycles             | 4          |
| Any Type Usages           | 138+       |
| Magic Numbers/Strings     | 100+       |
| Silent Exception Handlers | 50+        |
| Bare Except Clauses       | 306        |
+---------------------------+------------+
```

### Health Score by Module

```
Module                    Health    Issues
----------------------------------------
src/core/                 [====    ] 40%   Duplicates, cycles
src/data/                 [======  ] 60%   MTF gaps
src/models/               [=====   ] 50%   Missing adapters
src/validation/           [======= ] 70%   Good but unused
src/config/               [====    ] 40%   3 hierarchies
src/coordination/         [==      ] 20%   Full duplicate
src/feature_selection/    [==      ] 20%   Full duplicate
```

---

## Architecture Overview

### Current Module Structure

```
src/
+-- core/                    # Foundation types, contracts, utilities
|   +-- contracts/           # DataContract, ModelContract (DUPLICATES)
|   +-- coordination/        # DUPLICATE of src/coordination/
|   +-- types.py             # DataRank enum (DUPLICATE)
|   +-- constants.py         # MODEL_DATA_RANKS, MODEL_TO_FAMILY
|
+-- coordination/            # IDENTICAL to src/core/coordination/
|   +-- alignment.py         # 479 lines, same as core/coordination/
|   +-- timeframe_coordinator.py
|
+-- config/                  # Configuration management
|   +-- global_config.py     # UnifiedConfig
|   +-- pipeline/            # PipelineConfig
|   +-- training.py          # TrainerConfig
|
+-- data/
|   +-- adapters/            # 2D/3D adapters (4D MISSING)
|   +-- features/            # 162 feature implementations
|   +-- pipeline/            # 12-stage data pipeline
|   +-- labeling/            # Triple-barrier, meta-labeling
|
+-- models/
|   +-- boosting/            # XGB, LGBM, CatBoost
|   +-- neural/              # LSTM, GRU, TCN
|   +-- ensemble/            # Stacking, diversity metrics
|   +-- calibration/         # Isotonic, conformal
|   +-- config/              # ModelFamily enum (DUPLICATE, has 6 vs 5)
|
+-- feature_selection/       # NEAR-DUPLICATE of src/optimization/
+-- optimization/
|   +-- feature_selection/   # Same functionality, different path
|
+-- validation/              # Leakage, lookahead, statistical tests
```

### Clean Architecture Violations

```
                    DEPENDENCY DIRECTION VIOLATIONS
    +---------------------------------------------------------+
    |                                                          |
    |   OUTER LAYER                 INNER LAYER                |
    |   (Models, Adapters)          (Core, Contracts)          |
    |                                                          |
    |   +-----------------+         +-----------------+        |
    |   |   src/models/   |         |   src/core/     |        |
    |   |                 | <------ |                 |        |
    |   | config/data_    |         | constants.py    |        |
    |   | requirements.py |         | imports from    |        |
    |   |                 |         | models.config   |        |
    |   +-----------------+         +-----------------+        |
    |         |                           ^                    |
    |         |   VIOLATION: Core         |                    |
    |         |   imports from Models     |                    |
    |         +---------------------------+                    |
    |                                                          |
    +---------------------------------------------------------+
```

### Type Duplication Map

```
+------------------+-----------------------------------+-----------------------------------+
| Type             | Location 1                        | Location 2                        |
+------------------+-----------------------------------+-----------------------------------+
| DataRank         | src/core/types.py                 | src/core/contracts/data_contract.py|
|                  | (TABULAR_2D, SEQUENCE_3D,         | (Same values, different class)    |
|                  |  MULTI_TF_4D)                     |                                   |
+------------------+-----------------------------------+-----------------------------------+
| DataContract     | src/core/contracts/data_contract.py| src/core/data_contract.py        |
|                  | (428 lines, full implementation)  | (Likely older version)            |
+------------------+-----------------------------------+-----------------------------------+
| ModelFamily      | src/core/types.py                 | src/models/config/data_requirements.py|
|                  | 5 values: BOOSTING, CLASSICAL,    | 6 values: adds TRANSFORMER        |
|                  | NEURAL, ENSEMBLE, META_LEARNER    |                                   |
+------------------+-----------------------------------+-----------------------------------+
| AdapterResult    | Multiple locations                | Inconsistent field definitions    |
+------------------+-----------------------------------+-----------------------------------+
```

### Contract Consistency Issues

```
MODEL_CONTRACTS vs MODEL_DATA_REQUIREMENTS Conflict:
+------------+-------------------+---------------------+
| Model      | MODEL_CONTRACTS   | MODEL_DATA_REQS     |
+------------+-------------------+---------------------+
| PatchTST   | data_rank: 3D     | data_rank: 4D       |
| iTransformer| data_rank: 3D    | data_rank: 4D       |
| CatBoost   | max_features: 200 | max_features: 180   |
+------------+-------------------+---------------------+
```

---

## Data Flows

### Target Pipeline Architecture

```
                              ML FACTORY DATA FLOW
    +=========================================================================+
    |                                                                          |
    |  RAW DATA                                                                |
    |  +----------------+                                                      |
    |  | OHLCV 1-min    |                                                      |
    |  | Parquet        |                                                      |
    |  +-------+--------+                                                      |
    |          |                                                               |
    |          v                                                               |
    |  PHASE 1: INGESTION                                                      |
    |  +----------------+                                                      |
    |  | Validate       |                                                      |
    |  | Clean          |                                                      |
    |  | Standardize    |                                                      |
    |  +-------+--------+                                                      |
    |          |                                                               |
    |          v                                                               |
    |  PHASE 2: CANONICAL STORES                                               |
    |  +-----------------------------+    +-----------------------------+      |
    |  | ENGINEERED FEATURES         |    | RAW MTF OHLCV               |      |
    |  | ~180 indicators             |    | 9 timeframes x OHLCV        |      |
    |  | data/canonical/engineered/  |    | data/canonical/raw_mtf/     |      |
    |  | STATUS: EXISTS              |    | STATUS: NOT IMPLEMENTED     |      |
    |  +-------------+---------------+    +-------------+---------------+      |
    |                |                                  |                      |
    |                v                                  v                      |
    |  +-------------+---------------+    +-------------+---------------+      |
    |  | TABULAR ADAPTER (2D)        |    | MULTI-RES ADAPTER (4D)      |      |
    |  | STATUS: EXISTS              |    | STATUS: NOT IMPLEMENTED     |      |
    |  +-------------+---------------+    +-------------+---------------+      |
    |                |                                  |                      |
    |                v                                  v                      |
    |  +-------------+---------------+    +-------------+---------------+      |
    |  | BOOSTING MODELS             |    | TRANSFORMER MODELS          |      |
    |  | XGBoost, LightGBM, CatBoost |    | PatchTST, iTransformer      |      |
    |  | STATUS: WORKING             |    | STATUS: BLOCKED             |      |
    |  +-------------+---------------+    +-------------+---------------+      |
    |                |                                  |                      |
    |                +----------------+-----------------+                      |
    |                                 |                                        |
    |                                 v                                        |
    |  PHASE 5: STACKING ENSEMBLE                                              |
    |  +-----------------------------+                                         |
    |  | OOF Predictions             |                                         |
    |  | Meta-Learner Training       |                                         |
    |  | STATUS: WORKING (2D only)   |                                         |
    |  +-------------+---------------+                                         |
    |                |                                                         |
    |                v                                                         |
    |  PHASE 7: EVALUATION                                                     |
    |  +-----------------------------+                                         |
    |  | Backtest with costs         |                                         |
    |  | Financial metrics           |                                         |
    |  | STATUS: WORKING             |                                         |
    |  +-----------------------------+                                         |
    |                                                                          |
    +=========================================================================+
```

### Critical Data Flow Gaps

```
GAP 1: Raw MTF OHLCV Canonical Store
+------------------------------------------------------------------+
|  EXPECTED:                                                        |
|  data/canonical/raw_mtf/                                          |
|  +-- MES_1min_train.parquet                                       |
|  +-- MES_5min_train.parquet                                       |
|  +-- MES_10min_train.parquet                                      |
|  +-- ...                                                          |
|  +-- MES_60min_train.parquet  (9 files per split)                 |
|                                                                   |
|  ACTUAL: DIRECTORY DOES NOT EXIST                                 |
|                                                                   |
|  IMPACT: All 4D models (PatchTST, iTransformer, TFT, N-BEATS)     |
|          cannot be trained                                        |
+------------------------------------------------------------------+

GAP 2: MTF Generator to MultiStreamAdapter Handoff
+------------------------------------------------------------------+
|  CODE PATH:                                                       |
|  MTFGenerator.generate() -> additional_dfs parameter              |
|                                                                   |
|  EXPECTED: additional_dfs populated with higher TF DataFrames     |
|  ACTUAL: additional_dfs is NEVER populated (empty dict)           |
|                                                                   |
|  IMPACT: MultiStreamAdapter receives empty dict, cannot build 4D  |
+------------------------------------------------------------------+

GAP 3: 5-Dimension Optuna Optimization
+------------------------------------------------------------------+
|  DIMENSIONS:                                                      |
|  1. Triple Barrier Params     [ ] NOT IMPLEMENTED                 |
|  2. Feature Selection         [ ] NOT IMPLEMENTED                 |
|  3. Feature Parameters        [ ] NOT IMPLEMENTED                 |
|  4. Feature Timeframes        [ ] NOT IMPLEMENTED                 |
|  5. Model Hyperparameters     [X] IMPLEMENTED                     |
|                                                                   |
|  IMPACT: Only 20% of optimization capability active               |
+------------------------------------------------------------------+

GAP 4: FeatureSpec Artifact Flow
+------------------------------------------------------------------+
|  EXPECTED FLOW:                                                   |
|  Optuna Trial -> FeatureSpec JSON -> ModelBundle                  |
|                                                                   |
|  ACTUAL:                                                          |
|  - FeatureSpec dataclass defined but NEVER instantiated           |
|  - Optuna trials don't save feature selections                    |
|  - ModelBundle doesn't embed FeatureSpec                          |
|                                                                   |
|  IMPACT: Reproducibility broken - cannot recreate exact config    |
+------------------------------------------------------------------+
```

### Validation System Integration Status

```
+----+-------------------------+----------+------------------+
| #  | System                  | Built    | Pipeline Wired   |
+----+-------------------------+----------+------------------+
| 1  | Label Quality Scoring   | YES      | YES (Stage 6)    |
| 2  | Meta-Labeling           | YES      | YES (Stage 7)    |
| 3  | Probability Calibration | YES      | YES (Bundles)    |
| 4  | Conformal Prediction    | YES      | NO               |
| 5  | Ensemble Diversity      | YES      | NO               |
| 6  | Statistical Tests       | YES      | NO               |
| 7  | Deflated Sharpe         | YES      | NO               |
| 8  | Bootstrap CIs           | YES      | NO               |
| 9  | Regime Detection        | YES      | OPTIONAL         |
| 10 | Leakage Detection       | YES      | NO (should block)|
| 11 | Lookahead Audit         | YES      | NO (should block)|
| 12 | Session Handling        | YES      | OPTIONAL         |
| 13 | Drift Monitoring        | YES      | NO               |
+----+-------------------------+----------+------------------+
```

---

## Decision Trees

### Model Selection Path

```
                        MODEL SELECTION DECISION TREE

                              [Start]
                                 |
                                 v
                    +------------------------+
                    | What is the data rank? |
                    +------------------------+
                       /         |         \
                      /          |          \
                     v           v           v
                  [2D]        [3D]        [4D]
                   |           |           |
                   v           v           v
              +--------+   +--------+   +--------+
              |Boosting|   |Sequence|   |Multi-TF|
              +--------+   +--------+   +--------+
                   |           |           |
                   v           v           v
            +-----------+ +---------+ +-----------+
            | XGBoost   | | LSTM    | | PatchTST  |
            | LightGBM  | | GRU     | | iTransfmr |
            | CatBoost  | | TCN     | | TFT       |
            +-----------+ +---------+ +-----------+
                   |           |           |
                   v           v           v
              [WORKING]   [WORKING]   [BLOCKED]
                                           |
                                           v
                                    Missing 4D Adapter
                                    Missing Raw MTF Store
```

### Adapter Selection Logic

```
                        ADAPTER ROUTING DECISION TREE

                    +---------------------------+
                    | ModelContract.mtf_mode    |
                    +---------------------------+
                         /        |        \
                        /         |         \
                       v          v          v
                   [none]   [indicators]  [multi_stream]
                     |           |              |
                     v           v              v
              +-----------+ +-----------+ +-------------+
              | Load      | | Load      | | Load Raw    |
              | Engineered| | Engineered| | MTF OHLCV   |
              | (exclude  | | (with MTF | | (9 TFs)     |
              |  MTF cols)| |  columns) | |             |
              +-----------+ +-----------+ +-------------+
                     |           |              |
                     v           v              v
              +-----------+ +-----------+ +-------------+
              | Tabular   | | Tabular   | | Multi-Res   |
              | Adapter   | | OR        | | 4D Adapter  |
              | (2D)      | | Sequence  | | (4D)        |
              |           | | (3D)      | |             |
              +-----------+ +-----------+ +-------------+
                     |           |              |
                     v           v              v
               [EXISTS]    [EXISTS]     [NOT IMPLEMENTED]
```

### Error Handling Decision Points

```
                    CRITICAL ERROR HANDLING GAPS

    +----------------------------------------------------------------+
    |                                                                 |
    |  DataContract.validate_dataframe()                              |
    |  +-----------------------------+                                |
    |  | Returns: (bool, list[str])  |                                |
    |  | Raises: NOTHING             |  <-- PROBLEM                   |
    |  +-----------------------------+                                |
    |                 |                                               |
    |                 v                                               |
    |  +-----------------------------+                                |
    |  | Caller must check bool      |                                |
    |  | But callers DON'T CHECK     |  <-- PROBLEM                   |
    |  +-----------------------------+                                |
    |                                                                 |
    |  ModelContract.validate_data_contract()                         |
    |  +-----------------------------+                                |
    |  | Method EXISTS               |                                |
    |  | But NEVER CALLED            |  <-- PROBLEM                   |
    |  +-----------------------------+                                |
    |                                                                 |
    |  LeakageDetector.detect()                                       |
    |  +-----------------------------+                                |
    |  | Detects leakage             |                                |
    |  | Logs warning                |                                |
    |  | DOES NOT BLOCK TRAINING     |  <-- PROBLEM                   |
    |  +-----------------------------+                                |
    |                                                                 |
    +----------------------------------------------------------------+
```

---

## Refactoring Trajectory

### Phase Dependency Graph

```
                         REFACTORING PHASES

    Phase 0: Deduplication (No Dependencies)
    +------------------------------------------+
    | - Remove src/coordination/ duplicate     |
    | - Remove src/feature_selection/ duplicate|
    | - Consolidate DataRank enums             |
    | - Consolidate ModelFamily enums          |
    +------------------------------------------+
                         |
                         v
    Phase 1: Contract Enforcement (Depends on Phase 0)
    +------------------------------------------+
    | - Wire validate_data_contract() calls    |
    | - Make validate_dataframe() raise errors |
    | - Add pre-training validation hooks      |
    +------------------------------------------+
                         |
                         v
    Phase 2: 4D Infrastructure (Depends on Phase 1)
    +------------------------------------------+
    | - Implement Raw MTF OHLCV store          |
    | - Implement MultiResolution4DAdapter     |
    | - Wire MTF Generator -> Adapter handoff  |
    +------------------------------------------+
                         |
                         v
    Phase 3: 5-Dimension Optuna (Depends on Phase 2)
    +------------------------------------------+
    | - Implement barrier param optimization   |
    | - Implement feature selection in trials  |
    | - Implement FeatureSpec artifact flow    |
    +------------------------------------------+
                         |
                         v
    Phase 4: Validation Integration (Can parallelize with Phase 3)
    +------------------------------------------+
    | - Wire leakage detection to block train  |
    | - Wire lookahead audit to block train    |
    | - Integrate ensemble diversity           |
    | - Add deflated Sharpe after Optuna       |
    +------------------------------------------+
                         |
                         v
    Phase 5: Unified Entry Point
    +------------------------------------------+
    | - Consolidate config hierarchies         |
    | - Build MLFactory class                  |
    | - Create end-to-end notebook             |
    +------------------------------------------+
```

---

## Cleanup Phases

### Phase 0: Deduplication (Priority: CRITICAL, Effort: LOW)

**Objective:** Eliminate code duplication to establish single sources of truth.

| Task | Files Affected | Lines Changed | Risk |
|------|---------------|---------------|------|
| Delete `src/coordination/` | 2 files | -500 | LOW |
| Delete `src/feature_selection/` | ~10 files | -2000 | MEDIUM |
| Consolidate DataRank | 2 files | ~50 | LOW |
| Consolidate ModelFamily | 2 files | ~30 | LOW |
| Update all imports | ~50 files | ~100 | MEDIUM |

**Acceptance Criteria:**
```python
# Before (VIOLATION):
from src.coordination.alignment import align_to_anchor
from src.core.coordination.alignment import align_to_anchor  # Same!

# After (CORRECT):
from src.core.coordination.alignment import align_to_anchor
# src/coordination/ no longer exists
```

### Phase 1: Contract Enforcement (Priority: CRITICAL, Effort: MEDIUM)

**Objective:** Make contracts actually enforce data integrity.

| Task | Current Behavior | Target Behavior |
|------|-----------------|-----------------|
| `validate_dataframe()` | Returns bool | Raises `ContractViolationError` |
| `validate_data_contract()` | Never called | Called before every adapter |
| Leakage detection | Logs warning | Raises `LeakageDetectedError` |
| Lookahead audit | Logs warning | Raises `LookaheadBiasError` |

**Implementation Pattern:**
```python
# Before:
is_valid, issues = contract.validate_dataframe(df)
# Caller ignores is_valid

# After:
contract.validate_dataframe_strict(df)  # Raises on failure
# No need to check - exception guarantees validity
```

### Phase 2: 4D Infrastructure (Priority: HIGH, Effort: HIGH)

**Objective:** Enable 4D models (PatchTST, iTransformer, TFT, N-BEATS).

```
IMPLEMENTATION ORDER:
+------------------------------------------------------------------+
| Step 1: Raw MTF Store                                             |
| - Modify Phase 2 pipeline to output 9 parquet files per split    |
| - Ensure shift(1) anti-lookahead applied                         |
| - Verify alignment to anchor timeframe                           |
+------------------------------------------------------------------+
           |
           v
+------------------------------------------------------------------+
| Step 2: TimeSeriesDataContainer 4D Support                        |
| - Add shape validation for (N, 9, T, 4)                          |
| - Update slicing logic for 4D                                    |
| - Add tests for 4D container                                     |
+------------------------------------------------------------------+
           |
           v
+------------------------------------------------------------------+
| Step 3: MultiResolution4DAdapter                                  |
| - Extend BaseAdapter (currently does NOT)                        |
| - Load from raw_mtf/ directory                                   |
| - Window into (N, 9, T, 4) tensors                               |
| - Register in adapter registry                                   |
+------------------------------------------------------------------+
           |
           v
+------------------------------------------------------------------+
| Step 4: Model Implementations                                     |
| - PatchTST (priority: HIGH)                                      |
| - iTransformer (priority: HIGH)                                  |
| - TFT (priority: MEDIUM)                                         |
| - N-BEATS (priority: MEDIUM)                                     |
+------------------------------------------------------------------+
```

### Phase 3: 5-Dimension Optuna (Priority: HIGH, Effort: HIGH)

**Objective:** Implement full optimization across all dimensions.

```python
# TARGET OPTUNA OBJECTIVE STRUCTURE

def optuna_objective(trial, model_contract, raw_ohlcv, horizon):
    # DIMENSION 1: Triple Barrier Parameters
    barrier_params = {
        "profit_threshold": trial.suggest_float("profit_thresh", 0.005, 0.03),
        "loss_threshold": trial.suggest_float("loss_thresh", 0.003, 0.02),
        "max_holding_bars": trial.suggest_int("max_hold", horizon // 2, horizon * 2),
    }

    # DIMENSION 2: Feature Selection (from model's base set)
    base_features = model_contract.base_feature_set
    selected = [f for f in base_features
                if trial.suggest_categorical(f"use_{f}", [True, False])]

    # DIMENSION 3: Feature Parameters
    feature_params = {}
    if "rsi" in selected:
        feature_params["rsi_period"] = trial.suggest_int("rsi_period", 7, 28)

    # DIMENSION 4: Feature Timeframes
    feature_timeframes = {}
    for feature in selected:
        feature_timeframes[feature] = trial.suggest_categorical(
            f"{feature}_tf", [5, 15, 30, 60]
        )

    # DIMENSION 5: Model Hyperparameters (EXISTS)
    hyperparams = get_hyperparams_for_model(trial, model_contract.name)

    # Compute, train, evaluate
    labels = compute_triple_barrier(raw_ohlcv, horizon, **barrier_params)
    X = compute_features(raw_ohlcv, selected, feature_params, feature_timeframes)
    return cross_val_score_purged(model_contract, X, labels, hyperparams)
```

### Phase 4: Validation Integration (Priority: MEDIUM, Effort: MEDIUM)

**Objective:** Wire validation systems into training pipeline.

| System | Integration Point | Blocking Behavior |
|--------|------------------|-------------------|
| Leakage Detection | Pre-training hook | Raise if leakage detected |
| Lookahead Audit | Pre-training hook | Raise if lookahead found |
| Ensemble Diversity | Model selection | Warn if diversity < threshold |
| Deflated Sharpe | Post-Optuna | Gate deployment if DSR < 0.5 |
| Conformal Prediction | Inference | Include prediction sets |
| Bootstrap CIs | Financial reports | Add confidence intervals |

### Phase 5: Unified Entry Point (Priority: MEDIUM, Effort: MEDIUM)

**Objective:** Single `MLFactory.run(config)` entry point.

```
CONSOLIDATED CONFIG HIERARCHY:

BEFORE (Three separate configs):
+-- UnifiedConfig (src/config/global_config.py)
+-- PipelineConfig (src/config/pipeline/)
+-- TrainerConfig (src/config/training.py)

AFTER (Single config):
+-- ExperimentConfig
    +-- data: DataConfig
    +-- labeling: LabelingConfig
    +-- models: list[ModelConfig]
    +-- ensemble: EnsembleConfig
    +-- optimization: OptimizationConfig
    +-- output: OutputConfig
```

---

## What NOT to Do

### Anti-Pattern 1: Adding More Duplicate Files

```
WRONG:
  "I need a new alignment function"
  -> Create src/utils/alignment.py
  -> Now 3 copies of alignment logic

RIGHT:
  "I need a new alignment function"
  -> Add to src/core/coordination/alignment.py
  -> Single source of truth
```

### Anti-Pattern 2: Ignoring Contract Validation Results

```python
# WRONG: Check but don't act
is_valid, issues = contract.validate_dataframe(df)
# proceed anyway...

# RIGHT: Validate strictly
try:
    contract.validate_dataframe_strict(df)
except ContractViolationError as e:
    logger.error("Contract violated: %s", e)
    raise
```

### Anti-Pattern 3: Silent Exception Suppression

```python
# WRONG: Swallow errors
try:
    result = risky_operation()
except Exception:
    pass  # 306 instances in codebase!

# RIGHT: Handle or propagate
try:
    result = risky_operation()
except SpecificError as e:
    logger.warning("Known issue: %s, using fallback", e)
    result = fallback_value
```

### Anti-Pattern 4: Bypassing Adapters

```python
# WRONG: Direct data manipulation
X = df[feature_columns].values  # No contract validation!
model.fit(X, y)

# RIGHT: Use adapter chain
adapter = get_adapter_for_model(model_contract)
container = adapter.adapt(df, feature_columns)
# Container has validated contract
model.fit(container.X_train, container.y_train)
```

### Anti-Pattern 5: Adding Validation Without Blocking

```python
# WRONG: Detect but don't block
if leakage_detector.detect(df):
    logger.warning("Leakage detected!")  # Warning is not enough
model.fit(...)  # Training proceeds with leaky data!

# RIGHT: Detect and block
leakage_report = leakage_detector.detect(df)
if leakage_report.has_leakage:
    raise LeakageDetectedError(leakage_report)
```

### Anti-Pattern 6: Hardcoding Model-Specific Constants

```python
# WRONG: Magic numbers
if model_name == "catboost":
    max_features = 180  # Where does this come from?
elif model_name == "tcn":
    max_features = 150

# RIGHT: Use contracts
max_features = MODEL_CONTRACTS[model_name].max_features
# Single source of truth in contracts
```

### Anti-Pattern 7: Creating New Config Hierarchies

```python
# WRONG: Add another config class
@dataclass
class MyNewConfig:  # Now 4 config systems!
    param1: int

# RIGHT: Extend existing config
@dataclass
class ExperimentConfig:
    # ... existing fields
    my_new_section: MyNewSectionConfig  # Compose, don't duplicate
```

### Anti-Pattern 8: Skipping MTF Shift(1) Anti-Lookahead

```python
# WRONG: Use MTF data directly
higher_tf_features = compute_features(higher_tf_df)
merged = pd.merge_asof(anchor_df, higher_tf_features)
# In-progress bar data leaks!

# RIGHT: Apply shift(1)
higher_tf_features = compute_features(higher_tf_df)
higher_tf_features = higher_tf_features.shift(1)  # Use COMPLETED bars only
merged = pd.merge_asof(anchor_df, higher_tf_features)
```

---

## Summary

### Critical Blockers (Must Fix First)

| # | Blocker | Impact | Effort |
|---|---------|--------|--------|
| 1 | Duplicate modules | Import confusion, maintenance burden | LOW |
| 2 | Raw MTF OHLCV store missing | 4D models blocked | HIGH |
| 3 | 4D adapter missing | 4D models blocked | HIGH |
| 4 | Contract validation not enforced | Silent failures | MEDIUM |
| 5 | Leakage/lookahead don't block | Corrupted models | LOW |
| 6 | 5-Dimension Optuna incomplete | Suboptimal configs | HIGH |

### Implementation Priority Matrix

```
                     EFFORT
               LOW        HIGH
           +---------+---------+
      HIGH | Phase 0 | Phase 2 |
           | Dedup   | 4D Infra|
IMPACT     +---------+---------+
           | Phase 1 | Phase 3 |
      MED  | Contract| 5D Optuna|
           | Enforce | FeatureSpec|
           +---------+---------+
```

### Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Duplicate modules | 2 | 0 |
| Dead code functions | 588 | < 50 |
| Import cycles | 4 | 0 |
| Validation coverage | 30% | 100% |
| 4D models working | 0/6 | 6/6 |
| Optuna dimensions | 1/5 | 5/5 |

### Timeline Estimate

```
Phase 0 (Deduplication):     1-2 days
Phase 1 (Contracts):         2-3 days
Phase 2 (4D Infrastructure): 5-7 days
Phase 3 (5D Optuna):         5-7 days
Phase 4 (Validation):        3-4 days
Phase 5 (Unified Entry):     3-4 days
                            --------
Total:                      19-27 days
```

---

*Document generated from codebase analysis on 2026-01-23.*
*Next review scheduled after Phase 0 completion.*
