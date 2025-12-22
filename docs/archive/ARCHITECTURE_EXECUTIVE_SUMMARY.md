# Architecture Review: Executive Summary

**Date:** 2025-12-21
**Assessment:** Phase 1 Complete, Phase 2 Requires Infrastructure
**Overall Score:** Phase 1: 8.5/10 | Phase 2 Readiness: 4/10

---

## TL;DR

**Phase 1 (Data Preparation):** Production-ready with excellent modular architecture. Minor cleanup needed (5 files exceed line limits, config consolidation required).

**Phase 2 (Model Training):** Solid foundation but **missing critical infrastructure** - no PyTorch datasets, model abstractions, or training orchestration. Estimated 2-3 weeks to implement.

---

## Phase 1: Architectural Strengths

### What's Working Exceptionally Well

1. **Modular Package Design** ✅
   - `feature_scaler/` package: 7 modules, clean separation (core, scalers, validators)
   - `stage2_clean/` package: 4 modules, single-responsibility design
   - `features/` package: 11 modules for technical indicators
   - **Pattern**: Each package < 650 lines per file, clear public APIs via `__init__.py`

2. **Separation of Concerns** ✅
   - **Orchestration layer**: `pipeline/runner.py` (304 lines) - dependency resolution, checkpointing
   - **Domain layer**: `stages/` packages - business logic (labeling, features, cleaning)
   - **Configuration layer**: `config.py`, `pipeline_config.py` - validated settings
   - **No circular dependencies detected**

3. **Pipeline Orchestration** ✅
   - **Pattern**: Command + Dependency Graph
   - Stage registry with explicit dependencies
   - Fail-fast execution (required stages halt pipeline)
   - State persistence for resume capability
   - Artifact tracking with checksums (manifest system)

4. **Configuration Management** ✅
   - Symbol-specific barrier params (MES asymmetric, MGC symmetric)
   - Validation at boundaries (`validate_config()` checks PURGE_BARS >= max_bars)
   - Serialization support for reproducibility

5. **Data Integrity** ✅
   - Manifest system with SHA256 checksums
   - Stage provenance tracking
   - Run comparison for A/B testing

---

## Phase 1: Issues Requiring Attention

### Critical Issues

**None.** Phase 1 is production-ready.

### High Priority Clean-Up (Before Phase 2)

1. **File Size Violations** (5 files)
   ```
   988 lines  generate_report.py       → Split into reporting/ package
   920 lines  stage5_ga_optimize.py    → Split into ga_optimize/ package
   900 lines  stage8_validate.py       → Split into validation/ package
   780 lines  pipeline_cli.py          → Extract subcommands
   740 lines  stage1_ingest.py         → Split into ingest/ package
   ```
   **Effort:** 16 hours (4h per file)

2. **Configuration Duplication**
   - `config.py` (global constants) + `pipeline_config.py` (run config)
   - **Action:** Merge into `config/` package (constants.py, pipeline.py, barriers.py)
   **Effort:** 4 hours

3. **Stage Coupling**
   - `stage5_ga_optimize.py` imports `from stage4_labeling import triple_barrier_numba`
   - **Action:** Move to `utils/labeling.py` for shared access
   **Effort:** 1 hour

**Total Clean-Up Effort:** ~21 hours (3 days)

---

## Phase 2: Critical Infrastructure Gaps

### What's Missing (Must Implement)

#### 1. Data Loading Layer ❌
```python
# NEEDED: src/models/datasets/time_series.py
class TimeSeriesDataset(torch.utils.data.Dataset):
    """Sliding window dataset for temporal sequences."""
    # Handles sequence_length windows, forecast_horizon targets
    pass

# NEEDED: src/models/datasets/data_module.py
class FinancialDataModule(pl.LightningDataModule):
    """Loads train/val/test splits, creates DataLoaders."""
    pass
```

**Why Critical:** PyTorch models require Dataset/DataLoader abstractions. Current pipeline outputs parquet files, not torch-ready sequences.

**Effort:** 8 hours

---

#### 2. Model Abstraction Layer ❌
```python
# NEEDED: src/models/base/base_model.py
class BaseTimeSeriesModel(pl.LightningModule, ABC):
    """Abstract base for N-HiTS, TFT, PatchTST, LSTM."""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def training_step(self, batch, batch_idx):
        # Shared training logic
        pass
```

**Why Critical:** Enables consistent interface across model families. Required for ensemble methods.

**Effort:** 6 hours

---

#### 3. Model Registry (Factory Pattern) ❌
```python
# NEEDED: src/models/registry.py
class ModelRegistry:
    """Factory for model instantiation."""

    @classmethod
    def register(cls, name: str):
        # Decorator to register models
        pass

    @classmethod
    def get(cls, name: str, **kwargs):
        # Instantiate by name
        pass

# Usage:
@ModelRegistry.register("nhits")
class NHiTS(BaseTimeSeriesModel):
    pass
```

**Why Critical:** Decouples model selection from training code. Supports dynamic model loading.

**Effort:** 4 hours

---

#### 4. Training Orchestration ❌
```python
# NEEDED: src/models/trainer/trainer.py
class ModelTrainer:
    """Orchestrates model training, validation, testing."""

    def train_model(
        self,
        model_name: str,
        data_module: FinancialDataModule,
        config: ModelConfig
    ) -> TrainingResult:
        # 1. Get model from registry
        # 2. Setup callbacks (checkpointing, early stopping)
        # 3. Train with Lightning Trainer
        # 4. Save predictions for ensemble
        pass
```

**Why Critical:** Centralizes training logic. Prevents code duplication across model families.

**Effort:** 12 hours

---

#### 5. Experiment Tracking ❌
```python
# NEEDED: src/models/experiments/tracker.py
class ExperimentTracker:
    """Wrapper for MLflow/Weights & Biases."""

    def log_params(self, params: Dict):
        pass

    def log_metrics(self, metrics: Dict):
        pass

    def log_model(self, model, name: str):
        pass
```

**Why Critical:** Track hyperparameters, metrics, model versions across experiments.

**Effort:** 8 hours

---

### Recommended src/models/ Structure

```
src/models/
├── datasets/
│   ├── time_series.py ........... TimeSeriesDataset (sliding windows)
│   └── data_module.py ........... FinancialDataModule (Lightning)
│
├── base/
│   ├── base_model.py ............ BaseTimeSeriesModel (ABC)
│   └── base_config.py ........... ModelConfig (dataclass)
│
├── architectures/
│   ├── nhits.py ................. N-HiTS implementation
│   ├── tft.py ................... Temporal Fusion Transformer
│   ├── patchtst.py .............. PatchTST
│   └── lstm.py .................. Baseline LSTM
│
├── registry.py .................. ModelRegistry (factory)
│
├── trainer/
│   ├── trainer.py ............... ModelTrainer (orchestration)
│   ├── callbacks.py ............. Custom callbacks
│   └── metrics.py ............... Custom metrics (Sharpe, etc.)
│
├── experiments/
│   └── tracker.py ............... ExperimentTracker (MLflow)
│
└── ensemble/
    ├── stacking.py .............. Stacking ensemble
    └── voting.py ................ Voting ensemble
```

**Total Effort:** ~38 hours (5 days)

---

## Timeline to Phase 2 Readiness

### Week 1: Foundation (Critical Path)

**Days 1-2: Data Loading Layer**
- [ ] Implement `TimeSeriesDataset` (sliding window logic)
- [ ] Implement `FinancialDataModule` (Lightning wrapper)
- [ ] Unit tests for sequence generation

**Days 3-4: Model Abstractions**
- [ ] Create `BaseTimeSeriesModel` (abstract base)
- [ ] Implement `ModelRegistry` (factory pattern)
- [ ] Create `ModelConfig` dataclass

**Day 5: Clean-Up**
- [ ] Consolidate config.py + pipeline_config.py
- [ ] Fix stage4→stage5 coupling

**Deliverable:** Can instantiate models and load data ✅

---

### Week 2: Model Implementations

**Days 1-2: Baseline Models**
- [ ] Implement LSTM architecture
- [ ] Implement simple Transformer

**Days 3-5: Advanced Models**
- [ ] Implement N-HiTS
- [ ] OR integrate library (pytorch-forecasting, GluonTS)

**Deliverable:** 2+ model families registered ✅

---

### Week 3: Training Infrastructure

**Days 1-2: Trainer + Callbacks**
- [ ] Implement `ModelTrainer` orchestration
- [ ] Add checkpointing, early stopping callbacks
- [ ] Add prediction saving for ensemble

**Days 3-4: Experiment Tracking**
- [ ] Integrate MLflow or Weights & Biases
- [ ] Add hyperparameter logging
- [ ] Add metric tracking

**Day 5: Integration**
- [ ] Add `train_models` stage to pipeline
- [ ] End-to-end test: data → training → predictions
- [ ] Documentation

**Deliverable:** Full training pipeline operational ✅

---

## Architectural Patterns to Follow

### 1. Maintain Modular Design
- **Rule:** No file > 650 lines
- **Pattern:** Split large files into packages with clear responsibilities
- **Example:** `feature_scaler/` (7 modules) instead of 1729-line monolith

### 2. Dependency Injection
- **Pattern:** Pass config/dependencies at construction
- **Example:** `FeatureScaler(config=ScalerConfig(...))`
- **Avoid:** Global state, singletons

### 3. Factory Pattern for Models
- **Pattern:** `ModelRegistry.register()` decorator
- **Benefit:** Decouples model selection from training code
- **Example:** `model = ModelRegistry.get("nhits", **config)`

### 4. Fail-Fast Validation
- **Pattern:** Validate inputs at boundaries
- **Example:** `__post_init__` validation in dataclasses
- **Benefit:** Catch errors early, not mid-pipeline

### 5. Artifact Tracking
- **Pattern:** Manifest system for all outputs
- **Extend:** Add `ModelArtifact` class for weights, hyperparameters
- **Benefit:** Reproducibility, experiment comparison

---

## Key Decisions for Phase 2

### Decision 1: ML Framework
**Recommendation:** PyTorch + Lightning
- **Why:** Industry standard, excellent ecosystem (Hugging Face, timm)
- **Alternative:** TensorFlow/Keras (heavier, less flexible)

### Decision 2: Model Implementations
**Recommendation:** Use existing libraries where possible
- **pytorch-forecasting:** N-HiTS, TFT implementations
- **GluonTS:** Wide range of time series models
- **Custom:** Only when needed (e.g., domain-specific architectures)

### Decision 3: Experiment Tracking
**Recommendation:** MLflow
- **Why:** Open-source, self-hosted, integrates with PyTorch
- **Alternative:** Weights & Biases (better UI, requires account)

### Decision 4: Hyperparameter Tuning
**Recommendation:** Keep DEAP for barrier tuning, add Optuna for models
- **DEAP:** Already integrated for GA optimization (stage5)
- **Optuna:** Better for deep learning hyperparameters

---

## Risk Assessment

### Low Risk ✅
- **Phase 1 stability:** Production-ready, well-tested
- **Extensibility:** Stage registry supports new stages easily
- **Modularity:** Package design proven with feature_scaler, stage2_clean

### Medium Risk ⚠️
- **Timeline:** 3 weeks assumes no major blockers
- **Model performance:** May require multiple iterations to tune
- **Ensemble complexity:** Stacking/blending adds training overhead

### High Risk ❌
- **None identified.** Architecture is sound, implementation is straightforward.

---

## Success Metrics

### Phase 1 Completion ✅
- [x] Modular package design (feature_scaler, stage2_clean)
- [x] 650-line limit compliance (7 active files, 5 need refactor)
- [x] Dependency graph (no circular dependencies)
- [x] Artifact tracking (manifest system)
- [x] Configuration validation (PURGE_BARS, barrier params)

### Phase 2 Readiness (In Progress)
- [ ] TimeSeriesDataset + DataModule implemented
- [ ] BaseTimeSeriesModel + ModelRegistry implemented
- [ ] ModelTrainer orchestration implemented
- [ ] 2+ model families registered (LSTM, N-HiTS)
- [ ] Experiment tracking integrated (MLflow)
- [ ] End-to-end test: data → training → predictions

---

## Recommendations

### Do Immediately (Week 1)
1. ✅ **Create src/models/ package structure** (scaffolding)
2. ✅ **Implement TimeSeriesDataset** (critical path)
3. ✅ **Consolidate configuration** (reduces confusion)
4. ✅ **Fix stage coupling** (improves testability)

### Do During Phase 2 (Weeks 2-3)
1. **Refactor oversized files** (parallel work)
2. **Integrate experiment tracking** (MLflow)
3. **Add baseline models** (LSTM, Transformer)
4. **Implement ensemble infrastructure** (stacking)

### Do Later (Post-Phase 2)
1. Add Pydantic validation (nice to have)
2. Implement plugin architecture (extensibility)
3. Add parallel execution (performance)
4. Docker support (deployment)

---

## Final Recommendations

### For Phase 1 Clean-Up (3 days)
```bash
# Priority 1: Consolidate config
mv src/config.py src/config/constants.py
mv src/pipeline_config.py src/config/pipeline.py

# Priority 2: Fix coupling
mv src/stages/stage4_labeling.py::triple_barrier_numba → src/utils/labeling.py

# Priority 3: Refactor oversized files (parallel work)
# Can be done incrementally, doesn't block Phase 2
```

### For Phase 2 Infrastructure (2-3 weeks)
```bash
# Week 1: Foundation
mkdir -p src/models/{datasets,base,architectures,trainer,experiments,ensemble}
# Implement TimeSeriesDataset, ModelRegistry, BaseTimeSeriesModel

# Week 2: Models
# Implement LSTM, N-HiTS (or integrate pytorch-forecasting)

# Week 3: Training
# Implement ModelTrainer, integrate MLflow, end-to-end testing
```

---

## Conclusion

**Phase 1:** Architecturally sound, production-ready. Minor clean-up recommended but not blocking.

**Phase 2:** Requires ~70 hours of focused development to implement missing infrastructure. Timeline is achievable with clear scope.

**Overall Assessment:** ✅ **Excellent foundation for multi-model ensemble system.** Modular design patterns from Phase 1 (feature_scaler, stage2_clean packages) provide blueprint for Phase 2 model packages.

**Next Steps:** Begin Week 1 foundation work (TimeSeriesDataset, ModelRegistry) immediately. Phase 1 clean-up can proceed in parallel.

---

**Full detailed analysis:** `/home/jake/Desktop/Research/docs/ARCHITECTURE_REVIEW_PHASE1_PHASE2.md`
