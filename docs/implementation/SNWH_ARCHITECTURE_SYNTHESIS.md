# SNwH Architecture Synthesis: Comprehensive Gap Analysis & Implementation Order

**Document Version:** 1.0
**Date:** 2026-01-16
**Purpose:** Synthesize findings from 4 parallel investigations into a unified implementation plan for the SNwH (Unified Multi-Timeframe Model Factory).

---

## Executive Summary

The SNwH implementation plan defines 8 phases (0-8) to achieve a unified pipeline where **any model family can train on any timeframe**, and **heterogeneous ensembles work by default**. Four parallel investigations examined the Models/Registry, Pipeline Stages, Config System, and CV/OOF components.

**Critical Path Identified:** The implementation MUST proceed in this order:
1. **Phase 0** - Canonical Contracts (foundation for everything else)
2. **Phase 1** - Config Layer (TrainerConfig + UnifiedConfig extensions)
3. **Phase 2** - Adapter Architecture (data shape routing)
4. **Phase 3** - Timeframe Coordination (MTF alignment)
5. **Phase 4** - OOF Integrity (stacking prerequisite)
6. **Phase 5** - Feature Strategy Unification (wiring existing strategies.py)
7. **Phase 6** - Validation & Diagnostics (pre-flight checks)
8. **Phase 7** - Testing Strategy
9. **Phase 8** - Documentation

**Blocking Gaps (Must Fix First):**
1. Trainer loads SINGLE timeframe for ALL models (blocks heterogeneous ensembles)
2. OOF coverage mismatch between tabular (100%) and sequence models (100%-seq_len)
3. TrainerConfig missing per-model timeframe fields (primary_timeframe, mtf_mode, adapter_id)

---

## 1. Dependency Graph

```
                                    Phase 0
                                 Canonical Contracts
                                   (Foundation)
                                        |
                    +-------------------+-------------------+
                    |                                       |
              Phase 1                                 Phase 0.3
         Config Layer                             Artifact Safety
    (TrainerConfig extension)                      (Parallel)
                    |
        +-----------+-----------+
        |           |           |
    Phase 2     Phase 3     Phase 5
    Adapters    Timeframe   Feature
    (2D/3D/4D)  Coordinate  Strategy
        |           |           |
        +-----------+-----------+
                    |
                Phase 4
             OOF Integrity
           (Stacking Support)
                    |
                Phase 6
         Validation & Diagnostics
                    |
                Phase 7
           Testing Strategy
                    |
                Phase 8
            Documentation
```

### Critical Dependencies

| Component | Depends On | Blocks |
|-----------|------------|--------|
| TrainerConfig extension | Phase 0 contracts | Adapter routing, TF coordination |
| AdapterRegistry | TrainerConfig.adapter_id | Multi-shape training |
| TimeframeCoordinator | MTF config fields | Per-model TF loading |
| OOF Coverage Fix | Adapter alignment | Heterogeneous stacking |
| Feature Strategy Wiring | TrainerConfig.feature_strategy | Per-model feature selection |
| Pre-flight Validation | All Phase 0-4 | Safe ensemble building |

---

## 2. Gap Priority Matrix

### Severity Scoring
- **CRITICAL**: Blocks heterogeneous ensembles (primary goal)
- **HIGH**: Causes silent failures or data corruption
- **MEDIUM**: Functional but suboptimal
- **LOW**: Documentation/UX improvements

### Effort Scoring
- **S**: <1 day (config change, simple addition)
- **M**: 1-3 days (new class/module, moderate refactoring)
- **L**: 3-5 days (significant refactoring, cross-module changes)
- **XL**: >5 days (architectural changes, many touchpoints)

### Priority Matrix (Impact x Effort)

| Gap ID | Description | Severity | Effort | Priority Score | SNwH Phase |
|--------|-------------|----------|--------|----------------|------------|
| **GAP-001** | Trainer loads SINGLE TF for ALL models | CRITICAL | L | **P1** | Phase 3 |
| **GAP-002** | OOF coverage mismatch (tabular 100% vs sequence <100%) | CRITICAL | M | **P1** | Phase 4 |
| **GAP-003** | TrainerConfig missing primary_timeframe, mtf_mode, adapter_id | HIGH | M | **P2** | Phase 1 |
| **GAP-004** | ModelDataRequirements missing input_rank, feature_mode | HIGH | S | **P2** | Phase 0 |
| **GAP-005** | MODEL_FEATURE_STRATEGIES not integrated with trainer | HIGH | M | **P2** | Phase 5 |
| **GAP-006** | DEFAULT_MTF_TIMEFRAMES = 7 TFs (missing 1min, 5min) | HIGH | S | **P3** | Phase 3 |
| **GAP-007** | Pre-flight validation missing (ensemble validated AFTER OOF gen) | HIGH | M | **P3** | Phase 6 |
| **GAP-008** | Sequence OOF allow_lookback_outside=True (potential leakage) | MEDIUM | S | **P4** | Phase 4 |
| **GAP-009** | Feature selection not tracked with OOF predictions | MEDIUM | M | **P4** | Phase 4 |
| **GAP-010** | UnifiedConfig missing per-model config section | MEDIUM | M | **P4** | Phase 1 |
| **GAP-011** | patchtst.yaml says neural_optimal, strategies.py says raw OHLCV | MEDIUM | S | **P5** | Phase 5 |
| **GAP-012** | entropy.py exceeds 1258 lines (file size violation) | MEDIUM | M | **P5** | N/A |
| **GAP-013** | Walk-forward training mode not integrated with OOF | LOW | M | **P6** | Phase 4 |

---

## 3. Detailed Gap Analysis by Investigation Area

### 3.1 Models & Registry (Agent 1 Findings)

**File:** `src/models/registry.py`

**Current State:**
- ModelRegistry stores: `name`, `family`, `description`, `aliases`, `class`
- `get_model_info()` returns: `requires_scaling`, `requires_sequences`, `default_config`
- **Missing:** `primary_timeframe`, `mtf_mode`, `input_rank`, `feature_mode`

```python
# Current (line 125-131)
cls._metadata[name] = {
    "name": name,
    "family": family,
    "description": description,
    "aliases": aliases,
    "class": model_class.__name__,
}

# Required (SNwH Phase 0.2):
cls._metadata[name] = {
    "name": name,
    "family": family,
    "description": description,
    "aliases": aliases,
    "class": model_class.__name__,
    "input_rank": 2,  # or 3 or 4
    "feature_mode": "engineered",  # or "raw" or "hybrid"
    "mtf_mode": "indicators",  # or "none" or "multi_stream"
    "default_timeframe": "5min",
}
```

**File:** `src/models/config/data_requirements.py`

**Current State (ModelDataRequirements):**
```python
# Lines 47-83 - missing fields
@dataclass(frozen=True)
class ModelDataRequirements:
    model_name: str
    family: ModelFamily
    feature_set: str
    requires_scaling: bool = False
    scaler_type: ScalerType = ScalerType.NONE
    requires_sequences: bool = False
    sequence_length: int = 60
    max_features: int | None = None
    # ... more fields ...
    # MISSING: input_rank, feature_mode, mtf_mode, primary_timeframe
```

**File:** `src/models/training/trainer.py`

**Critical Issue (GAP-001) at lines 314-317:**
```python
# Current: Loads SINGLE container for ALL models
def run(self, container: TimeSeriesDataContainer, skip_save: bool = False):
    # ...
    X_train_df, y_train_series, w_train_series = container.get_sklearn_arrays(
        "train", return_df=True
    )
```

The trainer receives ONE container with ONE timeframe. For heterogeneous ensembles with models on different timeframes, the trainer needs to:
1. Receive multiple containers (one per timeframe), OR
2. Load timeframe-specific data per model dynamically

**Recommended Fix:** Modify `Trainer.run()` to accept per-model timeframe configuration and load appropriate data.

### 3.2 Pipeline Stages (Agent 2 Findings)

**File:** `src/phase1/stages/mtf/constants.py`

**Issue (GAP-006) at lines 42-50:**
```python
# Current: 7 TFs (missing 1min, 5min)
DEFAULT_MTF_TIMEFRAMES = normalize_timeframe_list([
    "10min",  # Short-term momentum
    "15min",  # Standard short-term
    "20min",  # Extended short-term
    "25min",  # Medium transition
    "30min",  # Standard medium-term
    "45min",  # Extended medium-term
    "60min",  # Hourly trend
])

# Required (9 TFs per CLAUDE.md):
DEFAULT_MTF_TIMEFRAMES = normalize_timeframe_list([
    "1min",   # Base timeframe
    "5min",   # Short-term
    "10min", "15min", "20min", "25min", "30min", "45min", "60min"
])
```

**File:** `src/features/strategies.py`

**Issue (GAP-005):** MODEL_FEATURE_STRATEGIES defined but NOT wired to trainer.

```python
# strategies.py defines per-model baseline features (lines 145-367)
MODEL_FEATURE_STRATEGIES = {
    "xgboost": ModelFeatureStrategy(
        baseline_features=BASELINE_MOMENTUM + ... + MTF_INDICATORS_1H,
        mtf_mode="indicators",
        ...
    ),
    "patchtst": ModelFeatureStrategy(
        baseline_features=BASELINE_PRICE + ["volume"],  # Raw OHLCV only
        mtf_mode="multi_stream",
        ...
    ),
}

# BUT trainer.py does NOT use this:
# - No import of MODEL_FEATURE_STRATEGIES
# - No call to get_strategy_for_model()
# - Feature set comes from TrainerConfig.feature_set (string like "boosting_optimal")
```

**Integration Point:** `src/models/training/features.py` - TrainerFeaturesMixin should call `get_strategy_for_model()`.

### 3.3 Config System (Agent 3 Findings)

**File:** `src/models/config/trainer_config.py`

**Issue (GAP-003):** TrainerConfig missing critical fields.

```python
# Current (lines 27-102)
@dataclass
class TrainerConfig:
    model_name: str
    horizon: int = 20
    feature_set: str = "boosting_optimal"
    sequence_length: int = 60
    batch_size: int = 256
    # ... training params ...

    # MISSING (Required for SNwH Phase 1):
    # primary_timeframe: str = "5min"
    # mtf_mode: str = "indicators"  # "none", "indicators", "multi_stream"
    # adapter_id: str = "tabular"   # "tabular", "sequence", "multi_stream"
    # feature_strategy: str = "baseline"  # "baseline", "optimized", "raw"
    # input_rank: int = 2  # 2, 3, or 4
```

**File:** `src/config/unified.py`

**Issue (GAP-010):** UnifiedConfig has no per-model section.

```python
# Current (lines 446-489): Global sections only
@dataclass
class UnifiedConfig:
    symbol: str = "MES"
    timeframes: TimeframesSection = ...
    training: TrainingSection = ...
    # ...

    # MISSING (Required for SNwH Phase 1.1):
    # models: dict[str, ModelSection] = {}  # Per-model overrides
    # Per ModelSection:
    #   - primary_timeframe
    #   - mtf_mode
    #   - feature_strategy
    #   - adapter_id
```

**Cross-File Inconsistency (GAP-011):**
- `config/models/patchtst.yaml`: `feature_set: neural_optimal`
- `src/features/strategies.py`: `patchtst.baseline_features = BASELINE_PRICE + ["volume"]` (raw OHLCV)

### 3.4 CV & OOF System (Agent 4 Findings)

**File:** `src/cross_validation/oof_sequence.py`

**Issue (GAP-002):** Coverage mismatch breaks stacking.

```python
# Lines 194-206: Sequence models have <100% coverage
coverage = float((~np.isnan(oof_preds)).mean())
# ...
# Each segment (symbol or gap-separated region) loses seq_len samples at start
expected_missing = n_segments * seq_len
expected_coverage = max(0.0, 1.0 - (expected_missing / n_samples))
```

**Problem:** Tabular OOF has 100% coverage, sequence OOF has ~(100% - seq_len/n_samples). When stacking, the meta-learner receives:
- Tabular OOF: [N samples]
- Sequence OOF: [N - seq_len samples] with NaN at start

**Current "fix" in stacking:** Drops rows with any NaN, losing valid tabular predictions.

**Issue (GAP-008):** Potential leakage at line 130-131:
```python
# allow_lookback_outside=True means sequence window can include data outside fold
train_result = seq_builder.build_fold_sequences(train_idx, allow_lookback_outside=True)
val_result = seq_builder.build_fold_sequences(val_idx, allow_lookback_outside=True)
```

If lookback window includes test data, this leaks future information into training sequences.

**Issue (GAP-007):** No pre-flight validation in `src/models/ensemble/stacking.py`. The ensemble validates compatibility AFTER hours of OOF generation, not before.

---

## 4. Implementation Order (Dependency-Aware)

### Sprint 1: Foundation (Week 1)

**Phase 0 Tasks:**

| Task | File(s) | Lines | Description | Effort |
|------|---------|-------|-------------|--------|
| 0.2-A | `src/models/config/data_requirements.py` | 47-83 | Add `input_rank`, `feature_mode`, `mtf_mode`, `primary_timeframe` to ModelDataRequirements | S |
| 0.2-B | `src/models/registry.py` | 125-131 | Extend metadata dict with new fields | S |
| 0.2-C | All model files (`boosting/*.py`, `neural/*.py`) | Various | Add class attributes for input_rank, mtf_mode | M |

**Phase 1 Tasks:**

| Task | File(s) | Lines | Description | Effort |
|------|---------|-------|-------------|--------|
| 1.1-A | `src/models/config/trainer_config.py` | 27-102 | Add `primary_timeframe`, `mtf_mode`, `adapter_id`, `feature_strategy` | S |
| 1.1-B | `src/config/unified.py` | 446-489 | Add `ModelConfigSection` and `models: dict[str, ModelConfigSection]` | M |
| 1.2 | New: `src/models/config/ensemble_resolver.py` | N/A | Create EnsemblePlan dataclass, expand ensemble aliases, validate compatibility | M |

### Sprint 2: Data Routing (Week 2)

**Phase 2 Tasks:**

| Task | File(s) | Lines | Description | Effort |
|------|---------|-------|-------------|--------|
| 2.1-A | New: `src/adapters/registry.py` | N/A | Create AdapterRegistry with TabularAdapter, SequenceAdapter, MultiStreamAdapter | M |
| 2.1-B | New: `src/adapters/base.py` | N/A | AdapterContract interface: `transform(canonical_data) -> model_input` | S |
| 2.2 | `src/models/training/trainer.py` | 253-511 | Integrate adapter routing based on model requirements | L |
| 2.3 | New: `src/adapters/validation.py` | N/A | Input signature validation (rank, dtype, feature count) | M |

**Phase 3 Tasks:**

| Task | File(s) | Lines | Description | Effort |
|------|---------|-------|-------------|--------|
| 3.1 | `src/phase1/stages/mtf/constants.py` | 42-50 | Fix DEFAULT_MTF_TIMEFRAMES to 9 TFs | S |
| 3.2 | New: `src/timeframes/coordinator.py` | N/A | TimeframeCoordinator class for alignment | M |
| 3.3 | `src/models/training/trainer.py` | 310-325 | Load per-model timeframe data | L |

### Sprint 3: OOF & Stacking (Week 3)

**Phase 4 Tasks:**

| Task | File(s) | Lines | Description | Effort |
|------|---------|-------|-------------|--------|
| 4.1 | `src/cross_validation/oof_sequence.py` | 54-241 | Fix coverage alignment (pad NaN to full length, use mask) | M |
| 4.2 | `src/cross_validation/oof_core.py` | Various | Add feature_selection metadata to OOF result | S |
| 4.3 | `src/models/ensemble/stacking.py` | Various | Handle NaN in OOF with masking instead of dropping | M |
| 4.4 | `src/cross_validation/oof_sequence.py` | 130-131 | Review allow_lookback_outside leakage risk | S |

**Phase 5 Tasks:**

| Task | File(s) | Lines | Description | Effort |
|------|---------|-------|-------------|--------|
| 5.1 | `src/models/training/features.py` | Various | Import and use MODEL_FEATURE_STRATEGIES | M |
| 5.2 | `src/features/strategies.py` | 145-367 | Sync with model YAML configs (resolve patchtst inconsistency) | S |
| 5.3 | `src/training/orchestrator.py` | 100-128 | Wire feature strategy to orchestrator | M |

### Sprint 4: Validation & Testing (Week 4)

**Phase 6 Tasks:**

| Task | File(s) | Lines | Description | Effort |
|------|---------|-------|-------------|--------|
| 6.1 | New: `src/validation/preflight.py` | N/A | Pre-flight validation before training | M |
| 6.2 | `src/models/ensemble/stacking.py` | fit() | Call preflight validation BEFORE OOF generation | S |
| 6.3 | New: `src/validation/input_signature.py` | N/A | Validate rank, dtype, shape against model requirements | M |

**Phase 7 Tasks:**

| Task | File(s) | Lines | Description | Effort |
|------|---------|-------|-------------|--------|
| 7.1 | New: `tests/test_adapters.py` | N/A | Unit tests for adapter shapes | M |
| 7.2 | New: `tests/test_heterogeneous_stacking.py` | N/A | Integration test: tabular + sequence + transformer | M |
| 7.3 | New: `tests/test_oof_coverage.py` | N/A | Regression test for OOF alignment | S |

---

## 5. Risk Assessment

### Silent Failures (High Risk - Undetected Corruption)

| Gap | Risk | Detection | Mitigation |
|-----|------|-----------|------------|
| GAP-002 (OOF mismatch) | Models trained on misaligned labels | None - produces bad model silently | Phase 4.1 fix + validation |
| GAP-008 (lookback leakage) | Inflated OOF metrics | None - looks like good performance | Review fold boundaries |
| GAP-011 (strategy mismatch) | Wrong features used | None - training proceeds | Sync strategies.py with YAMLs |

### Loud Crashes (Lower Risk - Obvious Failure)

| Gap | Crash Type | When | User Impact |
|-----|------------|------|-------------|
| GAP-001 (single TF) | Shape mismatch | Heterogeneous ensemble training | Clear error message |
| GAP-003 (missing config) | AttributeError | Accessing new fields | Add fields first |
| GAP-006 (7 vs 9 TFs) | FileNotFoundError | Loading 1min/5min data | Update constant |

### Mitigation Strategy

1. **Add pre-flight validation** (Phase 6) BEFORE expensive operations
2. **Add shape assertions** at adapter boundaries
3. **Log input signatures** for debugging
4. **Create canary tests** that fail on silent corruption

---

## 6. Integration Points

### Where Components Connect

```
UnifiedConfig.to_trainer_config()
        |
        v
TrainerConfig (with new fields)
        |
        v
+-------+-------+
|               |
v               v
ModelRegistry   TimeframeCoordinator
(input_rank,    (load per-model TF)
 mtf_mode)            |
        |             v
        v      TimeSeriesDataContainer
AdapterRegistry      (per-TF)
        |             |
        v             v
+-------+-------------+
|
v
Trainer.run()
        |
        +---------> FeaturesMixin.get_strategy()
        |                   |
        |                   v
        |           MODEL_FEATURE_STRATEGIES
        |
        +---------> AdapterRegistry.get_adapter()
        |                   |
        |                   v
        |           TabularAdapter / SequenceAdapter / MultiStreamAdapter
        |
        +---------> OOFGenerator (if ensemble)
                            |
                            v
                    StackingEnsemble.fit()
                            |
                            v
                    Meta-learner training
```

### File Touchpoints Per Phase

| Phase | Files Modified | Files Created |
|-------|----------------|---------------|
| 0 | data_requirements.py, registry.py, all model files | - |
| 1 | trainer_config.py, unified.py | ensemble_resolver.py |
| 2 | trainer.py | adapters/registry.py, adapters/base.py, adapters/validation.py |
| 3 | mtf/constants.py, trainer.py | timeframes/coordinator.py |
| 4 | oof_sequence.py, oof_core.py, stacking.py | - |
| 5 | training/features.py, strategies.py, orchestrator.py | - |
| 6 | stacking.py | validation/preflight.py, validation/input_signature.py |
| 7 | - | tests/test_adapters.py, tests/test_heterogeneous_stacking.py, tests/test_oof_coverage.py |

---

## 7. Success Criteria Mapping

| SNwH Criterion | Gaps to Fix | Phases Required |
|----------------|-------------|-----------------|
| Per-model timeframe config works | GAP-001, GAP-003, GAP-006 | 0, 1, 3 |
| Adapters convert canonical OHLCV to model shapes | GAP-004 | 0, 2 |
| Heterogeneous stacking works with mixed families | GAP-001, GAP-002, GAP-007 | 2, 3, 4, 6 |
| Ensemble validation fails fast | GAP-007 | 6 |
| OOF integrity enforced | GAP-002, GAP-008, GAP-009 | 4 |
| Reproducibility via config hash | GAP-009 | 0.3 |
| Single-contract isolation | Already working | - |

---

## 8. Recommended Next Steps

### Immediate (This Week)

1. **Fix GAP-006** (S effort): Update DEFAULT_MTF_TIMEFRAMES to 9 TFs
2. **Fix GAP-004** (S effort): Add missing fields to ModelDataRequirements
3. **Fix GAP-003** (M effort): Extend TrainerConfig with timeframe fields
4. **Create pre-flight validation skeleton** (GAP-007)

### Short-Term (Next 2 Weeks)

5. **Implement AdapterRegistry** (Phase 2)
6. **Implement TimeframeCoordinator** (Phase 3)
7. **Fix OOF coverage alignment** (GAP-002)

### Medium-Term (Next Month)

8. **Wire MODEL_FEATURE_STRATEGIES** (Phase 5)
9. **Complete validation layer** (Phase 6)
10. **Build comprehensive test suite** (Phase 7)

---

## Appendix A: File Reference Index

| File | Purpose | Key Lines |
|------|---------|-----------|
| `src/models/registry.py` | Model plugin system | 125-131 (metadata) |
| `src/models/config/data_requirements.py` | Model data contracts | 47-83 (dataclass) |
| `src/models/config/trainer_config.py` | Training configuration | 27-102 (dataclass) |
| `src/models/training/trainer.py` | Training orchestration | 253-511 (run method) |
| `src/models/ensemble/validator.py` | Ensemble compatibility | 37-131 (validation) |
| `src/models/ensemble/stacking.py` | Stacking implementation | fit() method |
| `src/config/unified.py` | Unified config system | 446-489 (dataclass) |
| `src/phase1/stages/mtf/constants.py` | MTF configuration | 42-50 (DEFAULT_MTF_TIMEFRAMES) |
| `src/features/strategies.py` | Per-model feature strategies | 145-367 (MODEL_FEATURE_STRATEGIES) |
| `src/cross_validation/oof_sequence.py` | Sequence OOF generation | 54-241 (generator) |
| `src/training/orchestrator.py` | Training orchestrator | 77-148 (run method) |

---

## Appendix B: Terminology

| Term | Definition |
|------|------------|
| **Heterogeneous Ensemble** | Ensemble with base models from different families (e.g., XGBoost + LSTM + PatchTST) |
| **Homogeneous Ensemble** | Ensemble with base models from same family (e.g., all boosting) |
| **Input Rank** | Dimensionality of model input: 2D (tabular), 3D (sequence), 4D (multi-stream) |
| **MTF Mode** | Multi-timeframe strategy: `none`, `indicators`, `multi_stream` |
| **OOF** | Out-of-Fold predictions used for stacking |
| **Primary Timeframe** | The timeframe a model trains on (e.g., 15min for CatBoost) |
| **Adapter** | Component that transforms canonical data to model-specific shape |
| **Pre-flight Validation** | Checks run BEFORE expensive operations (OOF generation, training) |

---

*Document generated by SNwH Architecture Synthesis Agent*
