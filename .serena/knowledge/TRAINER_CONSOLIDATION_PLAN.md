# Trainer Consolidation Plan

> Status: ANALYSIS COMPLETE
> Date: 2026-01-20
> Target: 4 trainers -> 1 unified trainer

---

## Executive Summary

The codebase currently has **4 separate trainer implementations** that have evolved over different project phases. This document analyzes each trainer, recommends a consolidation path, and outlines the migration steps needed to achieve a single unified trainer.

---

## Current State Analysis

### 1. `src/models/training/trainer.py` (786 lines) - **MAIN/CANONICAL**

**Purpose:** Core model training orchestrator using `TrainerConfig`.

**Architecture:**
- Modular design with mixins:
  - `TrainerFeaturesMixin`: Feature selection and feature set resolution
  - `TrainerEvaluationMixin`: Test set evaluation functionality
  - `TrainerArtifactsMixin`: Artifact saving (configs, metrics, models)
- Uses `TrainerConfig` from `src/models/config`
- Integrates with `ModelRegistry` for model instantiation
- Supports experiment tracking via `ExperimentTracker`

**Key Features:**
- Pipeline lineage validation (data integrity checking)
- Feature selection (MDA-based) for tabular/classical models
- Feature set filtering (predefined feature groups)
- Heterogeneous ensemble support (2D tabular + 3D sequence data)
- Probability calibration (isotonic/sigmoid)
- Multi-timeframe data loading via `TimeframeCoordinator`
- Checksum generation for artifacts
- Label end times for purging overlapping labels

**Input:** `TimeSeriesDataContainer`
**Output:** Dict with run_id, training_metrics, evaluation_metrics, test_metrics, output_path, feature_selection

**Importers (11 files):**
```
src/training/unified_orchestrator.py
src/training/regime_trainer.py
src/training/model_trainer.py
src/training/modes/regime_aware.py
src/training/orchestrator.py
src/training/modes/meta_labeling.py
scripts/train_meta_labeling.py
scripts/train_regime_aware.py
src/models/training/__init__.py
```

---

### 2. `src/training/model_trainer.py` (569 lines) - **PHASE_3 ADAPTER INTEGRATION**

**Purpose:** Bridges PHASE_2 adapters with the main Trainer. Uses `PipelineConfig` as single config source.

**Architecture:**
- `ModelTrainer` class wraps the main `Trainer`
- `TrainedModelArtifact` dataclass for results
- Uses `UnifiedDataPreparation` from `src/adapters`
- Handles different data ranks (2D/3D/4D) transparently

**Key Features:**
- Automatic adapter selection via `MODEL_DATA_RANKS` and `MODEL_ADAPTER_MAP`
- Optuna hyperparameter optimization integration
- `train_all()` for batch training multiple models
- `TrainedModelArtifact` with comprehensive metadata
- Training summary JSON export

**Input:** Raw `pd.DataFrame` (OHLCV with features/labels)
**Output:** `TrainedModelArtifact` or `Dict[str, TrainedModelArtifact]`

**Importers (3 files):**
```
src/training/__init__.py
scripts/phase3_validation.py
src/training/model_trainer.py (self)
```

**Relationship to Main Trainer:**
- **Internally uses** `src.models.Trainer` and `TrainerConfig`
- Acts as a higher-level orchestrator that:
  1. Prepares data via adapters
  2. Builds `TimeSeriesDataContainer`
  3. Delegates actual training to main `Trainer`

---

### 3. `src/training/regime_trainer.py` (792 lines) - **REGIME-AWARE SPECIALIZATION**

**Purpose:** Regime-specific model training using `PipelineConfig`. Detects market regimes and trains separate models per regime.

**Architecture:**
- `RegimeAwareTrainer` class
- `RegimeModelResult` and `RegimeTrainingResult` dataclasses
- Uses `RegimeDetector` from `src/training/regime_detector`
- Two training modes:
  1. Separate models per regime
  2. Single model with regime as one-hot feature

**Key Features:**
- Multiple regime detection methods (volatility_percentile, trend_adx, combined)
- Configurable number of regimes (2 or 3)
- Regime-specific data splitting
- Weighted metric aggregation across regimes
- Prediction routing based on detected regime

**Input:** `PreparedData` from PHASE_2 adapters
**Output:** `RegimeTrainingResult`

**Importers (2 files):**
```
src/training/unified_orchestrator.py
src/training/regime_trainer.py (self, for imports)
```

**Relationship to Main Trainer:**
- **Internally uses** `src.models.Trainer` and `TrainerConfig`
- Creates regime-specific `TimeSeriesDataContainer` instances
- Delegates actual training to main `Trainer`

---

### 4. `src/training/modes/regime_aware.py` (885 lines) - **LEGACY REGIME-AWARE**

**Purpose:** Older regime-aware training using `ExperimentConfig`. Similar functionality to `regime_trainer.py` but uses different config system.

**Architecture:**
- `RegimeAwareTrainer` class (different from regime_trainer.py)
- `RegimeAwareConfig`, `RegimeTrainingResult`, `RegimeAwareTrainingResult` dataclasses
- Built-in regime detection functions (not using `RegimeDetector`)
- Two training modes: separate models or regime-as-feature

**Key Features:**
- Three regime types: volatility, trend, composite
- ATR-based volatility detection
- ADX-based trend detection
- Composite regimes (4 combinations of trend x volatility)
- Uses `TimeSeriesDataContainer.from_dataframes()` for regime splits

**Input:** `TimeSeriesDataContainer` or loads from config path
**Output:** `Dict` with model_results, summary, output_path, total_time

**Importers (3 files):**
```
Not done yet/PHASE_3_TRAINING_ORCHESTRATION.md (docs only)
tests/test_training_modes.py
src/ml_pipeline/unified.py
```

**Relationship to Main Trainer:**
- **Internally uses** `src.models.Trainer` and `TrainerConfig`
- Delegates actual training to main `Trainer`

---

## Dependency Hierarchy

```
                           PipelineConfig
                                 |
                                 v
                    UnifiedTrainingOrchestrator
                         (src/training/unified_orchestrator.py)
                                 |
          +----------------------+----------------------+
          |                      |                      |
          v                      v                      v
    ModelTrainer           RegimeAwareTrainer     (walk_forward/meta_labeling)
 (model_trainer.py)     (regime_trainer.py)            (modes/)
          |                      |                      |
          +----------+-----------+----------+-----------+
                     |                      |
                     v                      v
                  Trainer              ExperimentConfig
         (models/training/trainer.py)        |
                     |                       v
                     +----> RegimeAwareTrainer (legacy)
                                (modes/regime_aware.py)
```

---

## Recommended Consolidation Path

### Canonical Trainer: `src/models/training/trainer.py`

**Rationale:**
1. **Most feature-complete**: Supports feature selection, calibration, heterogeneous ensembles, lineage validation
2. **Best architecture**: Modular mixin design enables extension without modification
3. **Core dependency**: All other trainers ultimately delegate to this one
4. **Proper abstraction**: Uses `TimeSeriesDataContainer` which is the established data interface
5. **Most maintained**: 786 lines of well-documented, production-ready code

### Consolidation Strategy

The other trainers are not truly redundant - they serve as **orchestration layers** that:
1. `ModelTrainer`: Integrates with PHASE_2 adapters and `PipelineConfig`
2. `RegimeAwareTrainer` (regime_trainer.py): Adds regime detection and splitting
3. `RegimeAwareTrainer` (modes/regime_aware.py): Legacy version for `ExperimentConfig`

The issue is **not that we have 4 trainers** but that we have:
- **2 competing config systems**: `TrainerConfig` vs `PipelineConfig` vs `ExperimentConfig`
- **2 competing regime trainers**: `regime_trainer.py` vs `modes/regime_aware.py`
- **Unclear entry points**: `ModelTrainer` vs `UnifiedTrainingOrchestrator` vs direct `Trainer`

---

## Migration Plan

### Phase 1: Consolidate Config Systems (PRIORITY: HIGH)

**Current State:**
- `TrainerConfig` (src/models/config/trainer_config.py) - used by main Trainer
- `PipelineConfig` (src/core/) - used by PHASE_3 orchestrators
- `ExperimentConfig` (src/training/config.py) - used by legacy code

**Action:**
1. Define mapping from `PipelineConfig` -> `TrainerConfig` (partially done in `model_trainer.py`)
2. Deprecate `ExperimentConfig` in favor of `PipelineConfig`
3. Update all orchestrators to convert `PipelineConfig` to `TrainerConfig` consistently

### Phase 2: Consolidate Regime Trainers (PRIORITY: HIGH)

**Current State:**
- `src/training/regime_trainer.py` - Uses `PipelineConfig`, `RegimeDetector`, `PreparedData`
- `src/training/modes/regime_aware.py` - Uses `ExperimentConfig`, built-in detection, `TimeSeriesDataContainer`

**Action:**
1. **Keep**: `src/training/regime_trainer.py` (modern, uses PipelineConfig)
2. **Deprecate**: `src/training/modes/regime_aware.py` (legacy, uses ExperimentConfig)
3. **Migrate**:
   - `tests/test_training_modes.py`: Update to use new `RegimeAwareTrainer`
   - `src/ml_pipeline/unified.py`: Update imports

**Unique Features to Preserve from Legacy:**
- Composite regime labels (`trending_low_vol`, etc.) - already in new version via `RegimeDetector`
- Direct container input - new version requires `PreparedData`, add adapter

### Phase 3: Clarify Entry Points (PRIORITY: MEDIUM)

**Recommended API:**

```python
# HIGH-LEVEL (recommended for most users)
from src.training import train_pipeline
result = train_pipeline(config, df)  # Uses UnifiedTrainingOrchestrator

# MID-LEVEL (for custom workflows)
from src.training import ModelTrainer
trainer = ModelTrainer(config)
artifact = trainer.train_model("xgboost", df, horizon=20)

# LOW-LEVEL (for maximum control)
from src.models import Trainer, TrainerConfig
trainer = Trainer(trainer_config)
results = trainer.run(container)
```

### Phase 4: Update Imports (PRIORITY: LOW)

Files requiring import updates after consolidation:

| File | Current Import | New Import |
|------|---------------|------------|
| `tests/test_training_modes.py` | `from src.training.modes.regime_aware import RegimeAwareTrainer` | `from src.training import RegimeAwareTrainer` |
| `src/ml_pipeline/unified.py` | `from src.training.modes.regime_aware import ...` | `from src.training import RegimeAwareTrainer` |

---

## Features to Merge into Canonical Trainer

### From `model_trainer.py`:
- [x] Already delegating to main Trainer - no merge needed
- [ ] `TrainedModelArtifact` dataclass could be useful as optional return type

### From `regime_trainer.py`:
- [x] Already delegating to main Trainer - no merge needed
- [ ] Consider adding regime-aware mode directly to main Trainer as optional feature

### From `modes/regime_aware.py` (to preserve before deprecation):
- [x] Regime detection already in `RegimeDetector` - no merge needed
- [ ] Direct `TimeSeriesDataContainer` input support (currently requires `PreparedData`)

---

## Files That Import Each Trainer

### Main Trainer (`src/models/training/trainer.py`)

**Direct imports via `from src.models import Trainer`:**
1. `src/training/unified_orchestrator.py` - Creates Trainer for each model
2. `src/training/regime_trainer.py` - Creates Trainer per regime
3. `src/training/model_trainer.py` - Creates Trainer per model
4. `src/training/modes/regime_aware.py` - Creates Trainer per regime
5. `src/training/orchestrator.py` - Legacy orchestrator
6. `src/training/modes/meta_labeling.py` - Meta-labeling training
7. `scripts/train_meta_labeling.py` - CLI script
8. `scripts/train_regime_aware.py` - CLI script

**Re-exported via `src/models/__init__.py` and `src/models/trainer.py`**

### ModelTrainer (`src/training/model_trainer.py`)

**Direct imports:**
1. `src/training/__init__.py` - Package re-export
2. `scripts/phase3_validation.py` - Validation script

### RegimeAwareTrainer (`src/training/regime_trainer.py`)

**Direct imports:**
1. `src/training/unified_orchestrator.py` - Regime-aware training mode
2. `src/training/__init__.py` - Package re-export

### Legacy RegimeAwareTrainer (`src/training/modes/regime_aware.py`)

**Direct imports:**
1. `tests/test_training_modes.py` - Unit tests
2. `src/ml_pipeline/unified.py` - ML pipeline integration

---

## Recommended Actions

### Immediate (This Week)
1. Add deprecation warning to `src/training/modes/regime_aware.py`
2. Document the intended entry points in `src/training/__init__.py` docstring
3. Add type annotations showing `PipelineConfig` -> `TrainerConfig` conversion

### Short Term (This Month)
1. Update `tests/test_training_modes.py` to use new `RegimeAwareTrainer`
2. Update `src/ml_pipeline/unified.py` to use new `RegimeAwareTrainer`
3. Add integration tests for all three API levels

### Medium Term (Next Quarter)
1. Remove `src/training/modes/regime_aware.py` after migration complete
2. Consolidate `ExperimentConfig` users to `PipelineConfig`
3. Consider adding regime-aware mode directly to main Trainer

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Breaking existing scripts | Medium | High | Add deprecation warnings first, maintain backward compat |
| Test failures | Low | Medium | Update tests incrementally |
| Documentation drift | Medium | Low | Update docs as part of migration |
| Config incompatibility | Medium | Medium | Thorough testing of config conversions |

---

## Conclusion

The trainer consolidation is primarily about:
1. **Config unification**: Converge on `PipelineConfig` for high-level API, `TrainerConfig` for low-level
2. **Regime trainer deduplication**: Keep `regime_trainer.py`, deprecate `modes/regime_aware.py`
3. **Clear API tiers**: `train_pipeline()` -> `ModelTrainer` -> `Trainer`

The main `Trainer` in `src/models/training/trainer.py` is already the canonical implementation - all other "trainers" are orchestration layers that delegate to it. The consolidation effort should focus on eliminating redundancy in the orchestration layer, not the core trainer.
