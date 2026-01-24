# Cleanup Plan: ML Factory

**Status:** Phase 0 Complete | Phase 1 Complete | Phase 2 Ready
**Generated:** 2026-01-23
**Last Updated:** 2026-01-23
**Lines Removed (Phase 0):** ~5,336
**Lines Added (Phase 1):** +616

---

## Table of Contents
- [Executive Summary](#executive-summary)
- [Completed: Phase 0](#completed-phase-0-deduplication)
- [Phase 1: Contract Enforcement](#phase-1-contract-enforcement-critical-medium-effort)
- [Phase 2: 4D Infrastructure](#phase-2-4d-infrastructure-high-high-effort)
- [Phase 3: 5-Dimension Optuna](#phase-3-5-dimension-optuna-high-high-effort)
- [Phase 4: Validation Integration](#phase-4-validation-integration-medium-medium-effort)
- [Phase 5: Unified Entry Point](#phase-5-unified-entry-point-medium-medium-effort)
- [Execution Roadmap](#execution-roadmap)
- [NOT Doing (Deferrals)](#not-doing-deferrals)
- [Validation Checklist](#validation-checklist)

---

## Executive Summary

The ML Factory codebase contains 449 Python files totaling ~85,000 lines of code. Analysis has identified 41 issues across four categories: architectural inconsistencies (11), code quality flaws (8), data flow gaps (8), and error handling gaps (14). This cleanup plan addresses the highest-impact issues in priority order.

### Priority Matrix

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

### Phase Dependencies

```
Phase 0: Deduplication ----+
                           |
                           v
Phase 1: Contract Enforcement ----+
                                  |
          +--- Phase 4 (parallel) |
          |                       v
          |    Phase 2: 4D Infrastructure ----+
          |                                   |
          |                                   v
          +-----> Phase 3: 5-Dimension Optuna ----+
                                                  |
                                                  v
                              Phase 5: Unified Entry Point
```

---

## Completed: Phase 0 Deduplication

**Status:** ✅ COMPLETE (2026-01-23)
**Lines Removed:** ~5,336
**Full Details:** See `X ( IN PROGRESS DOCS) X/PHASE_0_COMPLETION.md`

### Summary

| Task | Description | Lines | Status |
|------|-------------|-------|--------|
| 0A | DataRank → `src/core/types.py` | -15 | ✅ |
| 0B | ModelFamily + TRANSFORMER | -30 | ✅ |
| 0C | Delete `src/coordination/` | -1,166 | ✅ |
| 0D | Delete `src/feature_selection/` | -3,508 | ✅ |
| 0E | MultiResolution4DAdapter consolidation | -617 | ✅ |
| 0F | AdapterResult compatibility properties | ±0 | ✅ |
| 0G | Rename DataContract → OHLCVValidationSchema | ±0 | ✅ |

### Results

- **Duplicate directories:** 2 → 0 (eliminated)
- **Duplicate enums:** 3 → 0 (consolidated)
- **Import paths:** All canonical, with backward-compatible re-exports
- **Verification:** 3 parallel review agents + Task Agent 7 all APPROVED

---

## Completed: Phase 1 Contract Enforcement

**Priority:** CRITICAL
**Status:** ✅ COMPLETE (2026-01-23)
**Actual Time:** 1 day
**Blocked By:** Phase 0
**Commit:** 7f71b52

### Problem Statement

Contracts exist but are never enforced at runtime. The `DataContract.validate_dataframe()` method returns `(bool, list[str])` but callers ignore the return value. Validation systems detect issues but log warnings instead of blocking execution, allowing corrupted data to enter training.

### Architecture Analysis

**Current State:**
```
                         CONTRACT VALIDATION FLOW (BROKEN)

+----------------+    +-------------------+    +------------------+
| DataContract   |    | Caller Code       |    | Training         |
| .validate_     |--->| result = validate |    | Pipeline         |
| dataframe()    |    | # IGNORES result  |--->| Trains on        |
| returns (bool, |    | proceed_anyway()  |    | invalid data!    |
| list[str])     |    +-------------------+    +------------------+
+----------------+

+----------------+    +-------------------+    +------------------+
| LeakageDetect  |    | Detection Code    |    | Training         |
| .detect()      |--->| if leakage:       |    | Pipeline         |
|                |    |   logger.warn()   |--->| Trains anyway!   |
|                |    | # NO EXCEPTION    |    | Leaky model.     |
+----------------+    +-------------------+    +------------------+
```

**Target State:**
```
                         CONTRACT VALIDATION FLOW (FIXED)

+----------------+    +-------------------+    +------------------+
| DataContract   |    | Caller Code       |    | Training         |
| .validate_     |--->| contract.validate |--->| Pipeline         |
| dataframe_     |    | _strict(df)       |    | Only runs on     |
| strict()       |    | # RAISES on fail  |    | valid data       |
| raises Error   |    +-------------------+    +------------------+
+----------------+

+----------------+    +-------------------+    +------------------+
| LeakageDetect  |    | Pre-Train Hook    |    | Training         |
| .detect()      |--->| if leakage:       |    | Pipeline         |
|                |    |   raise Leakage   |-X->| BLOCKED          |
|                |    |   DetectedError() |    | No leaky models  |
+----------------+    +-------------------+    +------------------+
```

### Execution Roadmap

| Sub-Phase | Task | Priority | Est. Effort | Dependencies | Status |
|-----------|------|----------|-------------|--------------|--------|
| 1A | Make `DataContract.validate_dataframe()` raise `DataContractViolation` on failure | CRITICAL | 2 hr | Phase 0 | ✅ |
| 1B | Call `ModelContract.validate_data_contract()` in adapter load path | CRITICAL | 2 hr | 1A | ✅ |
| 1C | Add pre-training validation hook in `UnifiedTrainingOrchestrator` | HIGH | 3 hr | 1A, 1B | ✅ |
| 1D | Wire leakage detection to block training when issues found | CRITICAL | 2 hr | 1C | ✅ |
| 1E | Wire lookahead audit to block training when issues found | CRITICAL | 2 hr | 1C | ✅ |
| 1F | Add scaler fit verification (must be train split only) | HIGH | 1 hr | 1C | ✅ |
| 1G | Fix chronological splits to validate sorting BEFORE computing indices | MEDIUM | 1 hr | None | ✅ |

### Files to Modify

| File | Line Range | Changes Required |
|------|------------|------------------|
| `src/core/contracts/data_contract.py` | 195-224 | Add `validate_dataframe_strict()` that raises exception |
| `src/core/contracts/model_contract.py` | ~100-150 | Add `validate_data_contract()` call enforcement |
| `src/data/adapters/base.py` | load() | Add contract validation before returning data |
| `src/training/orchestrator.py` | pre_train() | Add validation hook calling leakage + lookahead |
| `src/validation/leakage_detection.py` | detect() | Add blocking mode parameter |
| `src/validation/lookahead_audit.py` | audit() | Add blocking mode parameter |
| `src/data/pipeline/stages/scaling/core.py` | fit() | Add split verification assertion |
| `src/data/pipeline/stages/splits/core.py` | split() | Validate chronological sort before indexing |

### Results

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Silent validation failures | 50+ | 0 | -100% ✅ |
| Leakage detection enforcement | 0% | 100% | +100% ✅ |
| Lookahead enforcement | 0% | 100% | +100% ✅ |
| Data corruption risk | HIGH | LOW | Eliminated ✅ |

### Implementation Pattern

```python
# BEFORE (current broken pattern):
is_valid, issues = contract.validate_dataframe(df)
# Caller ignores is_valid, proceeds anyway

# AFTER (enforced pattern):
class DataContractViolation(Exception):
    """Raised when data violates contract."""
    def __init__(self, issues: list[str]):
        self.issues = issues
        super().__init__(f"Contract violations: {issues}")

def validate_dataframe_strict(self, df: pd.DataFrame) -> None:
    """Validate DataFrame, raising on failure."""
    is_valid, issues = self.validate_dataframe(df)
    if not is_valid:
        raise DataContractViolation(issues)
```

---

## Phase 2: 4D Infrastructure (HIGH, HIGH EFFORT)

**Priority:** HIGH
**Status:** Analysis Complete
**Estimated Time:** 5-7 days
**Blocked By:** Phase 1

### Problem Statement

4D models (PatchTST, iTransformer, TFT, N-BEATS) cannot train due to missing infrastructure. The Raw MTF OHLCV canonical store does not exist, preventing `MultiResolution4DAdapter` from loading data. The adapter exists but does not extend `BaseAdapter` and is not registered in `AdapterRegistry`.

### Architecture Analysis

**Current State:**
```
                              4D MODEL TRAINING FLOW (BLOCKED)

+------------------+     +------------------+     +------------------+
| Pipeline Stage 2 |     | MTFGenerator     |     | 4D Adapter       |
| Produces:        |---->| additional_dfs   |---->| Expects:         |
| Engineered       |     | parameter is     |     | raw_mtf/ dir     |
| features only    |     | NEVER populated  |     | DOES NOT EXIST   |
+------------------+     | (empty dict)     |     +------------------+
                         +------------------+              |
                                                          v
                                               +------------------+
                                               | PatchTST,        |
                                               | iTransformer,    |
                                               | TFT, N-BEATS     |
                                               | STATUS: BLOCKED  |
                                               +------------------+

Missing Infrastructure:
+-- data/
    +-- canonical/
        +-- engineered/          # EXISTS - 180 feature indicators
        +-- raw_mtf/             # MISSING - 9 TF x OHLCV parquets
            +-- {symbol}_{tf}_{split}.parquet   # NEEDED
```

**Target State:**
```
                              4D MODEL TRAINING FLOW (WORKING)

+------------------+     +------------------+     +------------------+
| Pipeline Stage 2 |     | Raw MTF Store    |     | 4D Adapter       |
| Produces:        |---->| Saves 9 TFs:     |---->| Loads from:      |
| Engineered +     |     | 1m, 3m, 5m, 10m, |     | raw_mtf/ dir     |
| Raw MTF OHLCV    |     | 15m, 30m, 60m,   |     | Windows into     |
+------------------+     | 2h, 4h           |     | (N, 9, T, 4)     |
                         +------------------+     +------------------+
                                                          |
                                                          v
                                               +------------------+
                                               | PatchTST,        |
                                               | iTransformer,    |
                                               | TFT, N-BEATS     |
                                               | STATUS: WORKING  |
                                               +------------------+

New Infrastructure:
+-- data/
    +-- canonical/
        +-- engineered/          # EXISTS
        +-- raw_mtf/             # NEW
            +-- MES_1min_train.parquet
            +-- MES_1min_val.parquet
            +-- MES_1min_test.parquet
            +-- MES_5min_train.parquet
            +-- ... (27 files per symbol)
```

### Execution Roadmap

| Sub-Phase | Task | Priority | Est. Effort | Dependencies | Status |
|-----------|------|----------|-------------|--------------|--------|
| 2A | Create `src/data/store/raw_mtf_store.py` - persist raw OHLCV at 9 timeframes | CRITICAL | 4 hr | Phase 1 | Pending |
| 2B | Add pipeline stage to populate raw MTF store after data_cleaning | CRITICAL | 3 hr | 2A | Pending |
| 2C | Fix PatchTST/iTransformer contracts to `DataRank.MULTI_TF_4D` | HIGH | 1 hr | Phase 0 | Pending |
| 2D | Make `MultiResolution4DAdapter` extend `BaseAdapter` | CRITICAL | 2 hr | 2A | Pending |
| 2E | Register 4D adapter with `AdapterRegistry` | CRITICAL | 1 hr | 2D | Pending |
| 2F | Wire `UnifiedDataPreparation` to auto-load from raw MTF store when `mtf_mode=multi_stream` | HIGH | 3 hr | 2E | Pending |
| 2G | Update `TimeSeriesDataContainer.get_multi_resolution_4d()` to use new store | HIGH | 2 hr | 2F | Pending |

### Files to Modify

| File | Action | Changes Required |
|------|--------|------------------|
| `src/data/store/raw_mtf_store.py` | CREATE | New module for raw OHLCV storage per timeframe |
| `src/data/pipeline/stages/mtf/core.py` | MODIFY | Add raw MTF store population step |
| `src/core/contracts/model_contract.py` | MODIFY | Update PatchTST, iTransformer data_rank to 4 |
| `src/data/adapters/multi_resolution.py` | MODIFY | Extend BaseAdapter, implement load() |
| `src/data/adapters/registry.py` | MODIFY | Register MultiResolution4DAdapter |
| `src/data/adapters/preparation.py` | MODIFY | Add mtf_mode=multi_stream routing |
| `src/core/datasets/container.py` | MODIFY | Update get_multi_resolution_4d() implementation |

### Expected Impact

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| 4D models trainable | 0/4 | 4/4 | +100% |
| Multi-TF store exists | NO | YES | Implemented |
| Adapter registration | Missing | Complete | Fixed |
| Model diversity | 19 models | 23 models | +4 models |

### 4D Tensor Structure

```python
# Target output shape for 4D adapter
X_4d.shape = (n_samples, n_timeframes, seq_len, n_features)
#             e.g. (10000,    9,          60,      4)
#                           ^            ^        ^
#                           |            |        +-- OHLC (no volume for raw)
#                           |            +----------- sequence window
#                           +------------------------ 9 timeframes

# Timeframe ordering (ascending resolution)
TIMEFRAMES = ["1min", "3min", "5min", "10min", "15min", "30min", "60min", "2h", "4h"]
```

---

## Phase 3: 5-Dimension Optuna (HIGH, HIGH EFFORT)

**Priority:** HIGH
**Status:** Analysis Complete
**Estimated Time:** 5-7 days
**Blocked By:** Phase 2

### Problem Statement

Optuna only optimizes hyperparameters (1 of 5 dimensions), leaving 80% of optimization capability unused. The current implementation does not optimize triple barrier parameters, feature selection, feature parameters, or feature timeframes. Additionally, labels are generated once and shared across all models, preventing model-specific label optimization.

### Architecture Analysis

**Current State:**
```
                    OPTUNA OPTIMIZATION (CURRENT - 1 DIMENSION)

+------------------+
| Labels Generated |    Fixed for all models
| ONCE at pipeline |    No per-model optimization
| start            |
+------------------+
         |
         v
+------------------+     +------------------+
| Optuna Trial     |     | Dimensions       |
|                  |     |                  |
| Only optimizes:  |     | [X] Hyperparams  |  <-- 20% of potential
| - learning_rate  |     | [ ] Barriers     |
| - max_depth      |     | [ ] Features     |
| - n_estimators   |     | [ ] Feat params  |
|                  |     | [ ] Feat TFs     |
+------------------+     +------------------+
```

**Target State:**
```
                    OPTUNA OPTIMIZATION (TARGET - 5 DIMENSIONS)

+------------------+     +------------------+     +------------------+
| Optuna Trial     |     | Dimension 1      |     | Dimension 2      |
|                  |---->| Triple Barrier   |---->| Feature          |
| Per-trial:       |     | - profit_thresh  |     | Selection        |
| Labels computed  |     | - loss_thresh    |     | - from base set  |
| Model-specific   |     | - max_hold_bars  |     | - per model      |
+------------------+     +------------------+     +------------------+
                                                          |
         +------------------------------------------------+
         |
         v
+------------------+     +------------------+     +------------------+
| Dimension 3      |     | Dimension 4      |     | Dimension 5      |
| Feature          |---->| Feature          |---->| Model            |
| Parameters       |     | Timeframes       |     | Hyperparams      |
| - RSI period     |     | - per feature    |     | - learning_rate  |
| - ATR window     |     | - [5m,15m,30m]   |     | - max_depth      |
+------------------+     +------------------+     +------------------+
                                                          |
                                                          v
                                               +------------------+
                                               | FeatureSpec      |
                                               | Artifact         |
                                               | - Saved to disk  |
                                               | - Embedded in    |
                                               |   ModelBundle    |
                                               +------------------+
```

### Execution Roadmap

| Sub-Phase | Task | Priority | Est. Effort | Dependencies | Status |
|-----------|------|----------|-------------|--------------|--------|
| 3A | Create `FeatureSpec` dataclass with all 5 dimensions | CRITICAL | 2 hr | Phase 2 | Pending |
| 3B | Define `BASE_FEATURE_SETS` per model family | HIGH | 3 hr | 3A | Pending |
| 3C | Implement unified Optuna objective optimizing all 5 dimensions | CRITICAL | 8 hr | 3A, 3B | Pending |
| 3D | Move label generation INSIDE Optuna trial (labels become model-specific) | CRITICAL | 4 hr | 3C | Pending |
| 3E | Save FeatureSpec artifact to `experiments/{run_id}/feature_specs/` | HIGH | 2 hr | 3C | Pending |
| 3F | Embed FeatureSpec in ModelBundle for inference parity | HIGH | 2 hr | 3E | Pending |

### Files to Modify

| File | Action | Changes Required |
|------|--------|------------------|
| `src/core/contracts/feature_spec.py` | CREATE | FeatureSpec dataclass with 5 dimensions |
| `src/optimization/base_feature_sets.py` | CREATE | Per-model-family feature set definitions |
| `src/optimization/optuna_objective.py` | MODIFY | Add 5-dimension trial structure |
| `src/data/labeling/triple_barrier.py` | MODIFY | Accept barrier params from Optuna trial |
| `src/training/bundle.py` | MODIFY | Add FeatureSpec to ModelBundle |
| `src/inference/predictor.py` | MODIFY | Load FeatureSpec from bundle |

### Expected Impact

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Optuna dimensions | 1/5 | 5/5 | +400% |
| Optimization completeness | 20% | 100% | +80% |
| Label optimization | Shared | Per-model | Enabled |
| Reproducibility | Partial | Full | FeatureSpec captures all |

### FeatureSpec Dataclass

```python
@dataclass
class FeatureSpec:
    """Complete specification for model training - all 5 dimensions."""

    # Dimension 1: Triple Barrier Parameters
    profit_threshold: float  # e.g., 0.015
    loss_threshold: float    # e.g., 0.010
    max_holding_bars: int    # e.g., 120

    # Dimension 2: Feature Selection
    selected_features: list[str]  # e.g., ["rsi_14", "atr_20", "macd_signal"]

    # Dimension 3: Feature Parameters
    feature_params: dict[str, dict[str, Any]]
    # e.g., {"rsi": {"period": 14}, "atr": {"window": 20}}

    # Dimension 4: Feature Timeframes
    feature_timeframes: dict[str, str]
    # e.g., {"rsi_14": "15min", "atr_20": "5min"}

    # Dimension 5: Model Hyperparameters
    hyperparameters: dict[str, Any]
    # e.g., {"learning_rate": 0.01, "max_depth": 6}

    # Metadata
    model_name: str
    optuna_trial_id: int
    created_at: str
    schema_hash: str
```

---

## Phase 4: Validation Integration (MEDIUM, MEDIUM EFFORT)

**Priority:** MEDIUM
**Status:** Analysis Complete
**Estimated Time:** 3-4 days
**Blocked By:** Phase 1 (can parallelize with Phase 3)

### Problem Statement

Validation systems exist but are not integrated into the training pipeline. Leakage detection, lookahead audit, ensemble diversity, and statistical validation all exist as standalone modules but are never called during training.

### Architecture Analysis

**Current State:**
```
                         VALIDATION SYSTEMS (DISCONNECTED)

+------------------+     +------------------+     +------------------+
| Leakage          |     | Lookahead        |     | Ensemble         |
| Detection        |     | Audit            |     | Diversity        |
| src/validation/  |     | src/validation/  |     | src/validation/  |
| leakage_.py      |     | lookahead_.py    |     | diversity_.py    |
+------------------+     +------------------+     +------------------+
        |                        |                        |
        v                        v                        v
   [NOT CALLED]            [NOT CALLED]            [NOT CALLED]
```

**Target State:**
```
                         VALIDATION SYSTEMS (INTEGRATED)

+------------------+     +------------------+     +------------------+
| Leakage          |     | Lookahead        |     | Training         |
| Detection        |---->| Audit            |---->| Proceeds         |
| BLOCKS if found  |     | BLOCKS if found  |     | Only if valid    |
+------------------+     +------------------+     +------------------+

+------------------+     +------------------+     +------------------+
| Post-Training    |     | Ensemble         |     | Financial        |
| Deflated Sharpe  |---->| Diversity        |---->| Report           |
| Gates deployment |     | Warns if low     |     | Bootstrap CIs    |
+------------------+     +------------------+     +------------------+
```

### Execution Roadmap

| Sub-Phase | Task | Priority | Est. Effort | Dependencies | Status |
|-----------|------|----------|-------------|--------------|--------|
| 4A | Call leakage_detection in `PipelineRunner.run_validation()` | CRITICAL | 2 hr | Phase 1 | Pending |
| 4B | Call lookahead_audit in `PipelineRunner.run_validation()` | CRITICAL | 2 hr | Phase 1 | Pending |
| 4C | Integrate DiversityAnalyzer into `EnsembleService.build_ensemble()` | HIGH | 2 hr | None | Pending |
| 4D | Add DeflatedSharpeRatio validation after Optuna optimization | HIGH | 3 hr | Phase 3 | Pending |
| 4E | Add Bootstrap CIs to financial report generation | MEDIUM | 2 hr | None | Pending |
| 4F | Make calibration automatic step in orchestrator (based on config) | MEDIUM | 2 hr | None | Pending |
| 4G | Connect bet sizing to backtest position sizing | MEDIUM | 2 hr | 4F | Pending |

### Files to Modify

| File | Action | Changes Required |
|------|--------|------------------|
| `src/data/pipeline/runner.py` | MODIFY | Add leakage + lookahead calls in run_validation() |
| `src/models/ensemble/service.py` | MODIFY | Add diversity analysis before ensemble creation |
| `src/optimization/optuna_runner.py` | MODIFY | Add DSR validation post-optimization |
| `src/evaluation/financial_report.py` | MODIFY | Add bootstrap CI calculation |
| `src/training/orchestrator.py` | MODIFY | Add automatic calibration step |
| `src/inference/backtesting/position_sizer.py` | MODIFY | Connect to bet sizing module |

### Expected Impact

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Leakage enforcement | 0% | 100% | +100% |
| Lookahead enforcement | 0% | 100% | +100% |
| Ensemble diversity check | Manual | Automatic | Integrated |
| Statistical validation | Manual | Automatic | Integrated |
| Confidence intervals | Missing | Present | Added |

### Validation Integration Points

```python
# Pre-training validation (Phase 1C + 4A + 4B)
def pre_training_validation(df: pd.DataFrame, config: Config) -> None:
    """Block training if validation fails."""
    # Contract validation
    contract.validate_dataframe_strict(df)

    # Leakage detection
    leakage_report = LeakageDetector().detect(df)
    if leakage_report.has_leakage:
        raise LeakageDetectedError(leakage_report)

    # Lookahead audit
    lookahead_report = LookaheadAudit().audit(df)
    if lookahead_report.has_lookahead:
        raise LookaheadBiasError(lookahead_report)

# Post-training validation (Phase 4D)
def post_optimization_validation(study: optuna.Study, threshold: float = 0.5) -> None:
    """Gate deployment based on statistical validity."""
    dsr = DeflatedSharpeRatio().compute(study)
    if dsr < threshold:
        logger.warning(f"Deflated Sharpe {dsr:.3f} below threshold {threshold}")
        raise OverfittingWarning(f"DSR={dsr:.3f}")
```

---

## Phase 5: Unified Entry Point (MEDIUM, MEDIUM EFFORT)

**Priority:** MEDIUM
**Status:** Analysis Complete
**Estimated Time:** 3-4 days
**Blocked By:** Phase 3

### Problem Statement

No single factory entry point exists; Pipeline, Training, and Inference are disconnected systems. Users must manually coordinate multiple entry points, configurations, and artifact handoffs.

### Architecture Analysis

**Current State:**
```
                         DISCONNECTED ENTRY POINTS

+------------------+     +------------------+     +------------------+
| PipelineRunner   |     | Training         |     | Inference        |
| run_pipeline()   |     | Orchestrator     |     | Predictor        |
|                  |     | train()          |     | predict()        |
+------------------+     +------------------+     +------------------+
        |                        |                        |
        v                        v                        v
   PipelineConfig          TrainerConfig           InferenceConfig
        |                        |                        |
        +------------------------+------------------------+
                                 |
                                 v
                      User must manually coordinate
                      artifact paths, config sync,
                      and execution order
```

**Target State:**
```
                         UNIFIED ENTRY POINT

                    +------------------+
                    | MLFactory        |
                    | .run(config)     |
                    +------------------+
                            |
        +-------------------+-------------------+
        |                   |                   |
        v                   v                   v
+------------------+ +------------------+ +------------------+
| Pipeline Stage   | | Training Stage   | | Inference Stage  |
| (automatic)      | | (automatic)      | | (automatic)      |
+------------------+ +------------------+ +------------------+
        |                   |                   |
        +-------------------+-------------------+
                            |
                            v
                    +------------------+
                    | ExperimentConfig |
                    | (single source   |
                    |  of truth)       |
                    +------------------+
```

### Execution Roadmap

| Sub-Phase | Task | Priority | Est. Effort | Dependencies | Status |
|-----------|------|----------|-------------|--------------|--------|
| 5A | Create `MLFactory` class that coordinates Pipeline -> Training -> Inference | CRITICAL | 6 hr | Phase 3 | Pending |
| 5B | Consolidate config hierarchies (`ExperimentConfig` as single source of truth) | HIGH | 4 hr | 5A | Pending |
| 5C | Create unified deployment bundle (`inference_bundle.tar.gz`) | HIGH | 3 hr | 5B | Pending |
| 5D | Remove deprecated `TrainingOrchestrator` | MEDIUM | 1 hr | 5A | Pending |
| 5E | Add Evaluation as explicit pipeline stage | MEDIUM | 2 hr | 5A | Pending |
| 5F | Write end-to-end Colab notebook | LOW | 3 hr | 5E | Pending |

### Files to Modify

| File | Action | Changes Required |
|------|--------|------------------|
| `src/factory.py` | CREATE | MLFactory class with run() method |
| `src/config/experiment.py` | CREATE | ExperimentConfig dataclass |
| `src/inference/bundle.py` | MODIFY | Add bundle packaging to tar.gz |
| `src/training/orchestrator_v1.py` | DELETE | Remove deprecated orchestrator |
| `src/data/pipeline/stages/evaluation/` | CREATE | New evaluation stage |
| `notebooks/end_to_end.ipynb` | CREATE | Comprehensive Colab notebook |

### Expected Impact

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Entry points | 3+ | 1 | -67% |
| Config classes | 3+ | 1 | Unified |
| User complexity | HIGH | LOW | Simplified |
| Onboarding time | Days | Hours | Reduced |

### MLFactory Interface

```python
class MLFactory:
    """Unified entry point for ML Factory operations."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self._pipeline = PipelineRunner(config.data)
        self._trainer = UnifiedTrainingOrchestrator(config.training)
        self._evaluator = Evaluator(config.evaluation)

    def run(self) -> ExperimentResult:
        """Execute full pipeline: data -> training -> evaluation."""
        # Phase 1: Data Pipeline
        data = self._pipeline.run()

        # Phase 2: Training
        models = self._trainer.train(data)

        # Phase 3: Evaluation
        results = self._evaluator.evaluate(models, data.test)

        # Phase 4: Bundle
        bundle = self._create_bundle(models, results)

        return ExperimentResult(
            models=models,
            metrics=results,
            bundle_path=bundle.path
        )

    def _create_bundle(self, models, results) -> DeploymentBundle:
        """Create deployment-ready bundle."""
        return DeploymentBundle.create(
            models=models,
            feature_spec=self._trainer.feature_spec,
            config=self.config,
            metrics=results
        )
```

---

## Execution Roadmap

### Timeline Overview

```
Week 1:
+-------+-------+-------+-------+-------+
| Mon   | Tue   | Wed   | Thu   | Fri   |
+-------+-------+-------+-------+-------+
| P0-A  | P0-D  | P0-E  | P1-A  | P1-B  |
| P0-B  | P0-D  | P0-F  | P1-A  | P1-C  |
| P0-C  |       | P0-G  |       |       |
+-------+-------+-------+-------+-------+
         Phase 0          Phase 1 starts

Week 2:
+-------+-------+-------+-------+-------+
| Mon   | Tue   | Wed   | Thu   | Fri   |
+-------+-------+-------+-------+-------+
| P1-C  | P1-D  | P1-F  | P2-A  | P2-B  |
| P1-C  | P1-E  | P1-G  | P2-A  | P2-C  |
|       |       |       |       | P2-D  |
+-------+-------+-------+-------+-------+
    Phase 1 complete     Phase 2 starts

Week 3:
+-------+-------+-------+-------+-------+
| Mon   | Tue   | Wed   | Thu   | Fri   |
+-------+-------+-------+-------+-------+
| P2-E  | P2-F  | P2-G  | P3-A  | P3-B  |
| P2-E  | P2-F  |       | P3-A  | P3-B  |
+-------+-------+-------+-------+-------+
  Phase 2 complete       Phase 3 starts

Week 4:
+-------+-------+-------+-------+-------+
| Mon   | Tue   | Wed   | Thu   | Fri   |
+-------+-------+-------+-------+-------+
| P3-C  | P3-C  | P3-D  | P3-E  | P4-A  |
| P3-C  | P3-C  | P3-D  | P3-F  | P4-B  |
+-------+-------+-------+-------+-------+
    Phase 3 continues      Phase 4 starts (parallel)

Week 5:
+-------+-------+-------+-------+-------+
| Mon   | Tue   | Wed   | Thu   | Fri   |
+-------+-------+-------+-------+-------+
| P4-C  | P4-D  | P4-F  | P5-A  | P5-B  |
| P4-C  | P4-E  | P4-G  | P5-A  | P5-C  |
+-------+-------+-------+-------+-------+
  Phase 4 complete       Phase 5 starts

Week 6:
+-------+-------+-------+
| Mon   | Tue   | Wed   |
+-------+-------+-------+
| P5-D  | P5-E  | P5-F  |
| P5-D  | P5-E  | DONE  |
+-------+-------+-------+
     Phase 5 complete
```

### Summary Table

| Phase | Description | Lines Changed | Effort | Dependencies |
|-------|-------------|---------------|--------|--------------|
| 0 | Deduplication | -2,000 | 1-2 days | None |
| 1 | Contract Enforcement | +500 | 2-3 days | Phase 0 |
| 2 | 4D Infrastructure | +1,500 | 5-7 days | Phase 1 |
| 3 | 5-Dimension Optuna | +2,000 | 5-7 days | Phase 2 |
| 4 | Validation Integration | +800 | 3-4 days | Phase 1 (parallel with 3) |
| 5 | Unified Entry Point | +1,200 | 3-4 days | Phase 3 |
| **Total** | | ~4,000 net | 19-27 days | |

---

## NOT Doing (Deferrals)

### 1. Refactoring 562 Long Functions

**Reason:** Low ROI - code works correctly, refactoring is high effort with minimal benefit.
- Functions work and are tested
- Refactoring would take weeks
- No immediate functional benefit
- Address incrementally during other work

### 2. Removing All 588 Dead Code Functions

**Reason:** Need API audit first - some may be used via dynamic dispatch.
- Some functions used via `getattr()` patterns
- Some are entry points for CLI commands
- Need comprehensive usage analysis
- Create deprecation warnings before removal

### 3. Eliminating All 138 Any Types

**Reason:** Gradual improvement during other work is more practical.
- Most `Any` types are in third-party interfaces
- Some are genuinely dynamic (plugin systems)
- Fix incrementally when touching files
- Prioritize new code being typed

### 4. Fixing All 100+ Magic Numbers

**Reason:** Create constants as encountered during other work.
- Many are in test files (acceptable)
- Some are domain-specific (ATR period = 14)
- Extract to constants when modifying code
- Not worth a dedicated refactoring pass

### 5. Removing All 306 Bare Except Clauses

**Reason:** High risk of breaking error handling - needs careful analysis.
- Some are intentional catch-alls
- Some protect against third-party exceptions
- Need case-by-case analysis
- Fix incrementally with testing

---

## Validation Checklist

### Per-Phase Validation

```bash
# After each phase, run:
ruff check src/                      # Linting
mypy src/ --ignore-missing-imports   # Type checking
pytest tests/ -x --tb=short          # Unit tests
python -c "import src; print('OK')"  # Import check
```

### Phase-Specific Checks

#### Phase 0
- [ ] No duplicate directories exist
- [ ] Single DataRank definition in `src/core/types.py`
- [ ] Single ModelFamily definition in `src/core/types.py`
- [ ] All imports resolve correctly

#### Phase 1
- [ ] `DataContract.validate_dataframe_strict()` raises on failure
- [ ] Leakage detection blocks training when issues found
- [ ] Lookahead audit blocks training when issues found
- [ ] Scaler fit only uses train split

#### Phase 2
- [ ] Raw MTF store populates during pipeline
- [ ] 4D adapter loads data correctly
- [ ] PatchTST training completes
- [ ] iTransformer training completes

#### Phase 3
- [ ] FeatureSpec dataclass exists
- [ ] Optuna optimizes all 5 dimensions
- [ ] FeatureSpec saved to disk
- [ ] ModelBundle contains FeatureSpec

#### Phase 4
- [ ] Leakage detection called in pipeline
- [ ] Lookahead audit called in pipeline
- [ ] Ensemble diversity analyzed
- [ ] Financial report includes CIs

#### Phase 5
- [ ] MLFactory.run() executes successfully
- [ ] Single ExperimentConfig works
- [ ] Deployment bundle created
- [ ] Colab notebook runs end-to-end

---

## Notes

### Implementation Order Rationale

Phase 0 (Deduplication) must come first because later phases import from consolidated locations. Phase 1 (Contracts) depends on clean imports. Phase 2 (4D) depends on contract enforcement. Phase 3 (Optuna) depends on 4D infrastructure. Phase 4 (Validation) can parallelize with Phase 3 since it only depends on Phase 1. Phase 5 (Unified) requires all other phases.

### Risk Mitigation

- Create feature branches for each phase
- Run full test suite after each sub-phase
- Keep old code commented (not deleted) until validation passes
- Document all breaking changes in CHANGELOG

### Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Duplicate modules | 2 | 0 |
| Import cycles | 4 | 0 |
| Validation coverage | 30% | 100% |
| 4D models working | 0/4 | 4/4 |
| Optuna dimensions | 1/5 | 5/5 |
| Entry points | 3+ | 1 |

---

*Document generated from codebase analysis on 2026-01-23.*
*Next review scheduled after Phase 0 completion.*
