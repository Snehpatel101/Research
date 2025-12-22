# Architecture Review: Phase 1 Completion & Phase 2 Readiness

**Date:** 2025-12-21
**Reviewer:** Claude Sonnet 4.5 (Architecture Specialist)
**Scope:** ML Pipeline Codebase - Phase 1 Data Preparation & Phase 2 Model Training Readiness

---

## Executive Summary

**Overall Architecture Score: 8.5/10** (Production-Ready with Minor Enhancements Needed)

The codebase demonstrates **excellent adherence to modular architecture principles** with strong separation of concerns, well-defined boundaries, and consistent design patterns. Phase 1 is architecturally complete and production-ready. Phase 2 requires specific infrastructure additions but has a solid foundation.

### Key Strengths
- Modular package architecture with clean boundaries (stage2_clean, feature_scaler)
- Excellent compliance with 650-line limit (7 files exceed, but 3 are archived _old.py)
- Strong separation: pipeline orchestration vs. domain logic vs. utilities
- Configuration-driven design with validation at boundaries
- Clear dependency direction (no circular dependencies detected)
- Artifact tracking and manifest system for reproducibility

### Critical Gaps for Phase 2
- **No PyTorch/Lightning infrastructure** (TimeSeriesDataset, DataModule, ModelRegistry)
- **No model abstraction layer** (base classes for N-HiTS, TFT, PatchTST, etc.)
- **No training orchestration** (fit/validate/test loops, checkpointing)
- **No experiment tracking** (MLflow, Weights & Biases integration)

---

## 1. Architecture Patterns Analysis

### 1.1 Current Architecture: Layered + Modular

```
┌─────────────────────────────────────────────────────────────────┐
│                     CLI Layer (pipeline_cli.py)                  │
└────────────────────────────────┬────────────────────────────────┘
                                 │
┌────────────────────────────────▼────────────────────────────────┐
│              Orchestration Layer (pipeline/runner.py)            │
│  - PipelineRunner: Stage execution, dependency resolution        │
│  - Manifest tracking, state persistence                          │
└────────────────────────────────┬────────────────────────────────┘
                                 │
┌────────────────────────────────▼────────────────────────────────┐
│         Stage Wrappers (pipeline/stages/*.py)                    │
│  - Thin adapters: run_data_cleaning(), run_feature_engineering()│
│  - Convert config → domain objects → StageResult                 │
└────────────────────────────────┬────────────────────────────────┘
                                 │
┌────────────────────────────────▼────────────────────────────────┐
│          Domain Logic (stages/*.py and packages)                 │
│  - stage2_clean/ package (DataCleaner, utils)                    │
│  - feature_scaler/ package (FeatureScaler, validators)           │
│  - features/ package (FeatureEngineer, momentum, volatility)     │
│  - stage4_labeling.py (triple-barrier logic)                     │
│  - stage5_ga_optimize.py (DEAP optimization)                     │
└────────────────────────────────┬────────────────────────────────┘
                                 │
┌────────────────────────────────▼────────────────────────────────┐
│      Configuration & Utilities (config.py, manifest.py)          │
│  - PipelineConfig (dataclass with validation)                    │
│  - Symbol-specific barrier params (MES vs MGC)                   │
│  - ArtifactManifest (checksums, versioning)                      │
└──────────────────────────────────────────────────────────────────┘
```

**Verdict:** ✅ Excellent separation of concerns. Each layer has clear responsibility.

---

### 1.2 Modular Package Design (Example: feature_scaler)

```
feature_scaler/
├── __init__.py ................. Public API exports
├── core.py ..................... Data classes, enums (ScalerType, FeatureCategory)
├── scalers.py .................. Utility functions (create_scaler, categorize_feature)
├── scaler.py ................... Main FeatureScaler class (546 lines)
├── validators.py ............... Leakage detection, validation logic
└── convenience.py .............. High-level APIs (scale_splits)
```

**Design Strengths:**
- **Single Responsibility:** Each module has one clear purpose
- **Dependency Injection:** Scaler type/config passed at construction
- **Interface Segregation:** Simple `scale_splits()` for common use, `FeatureScaler` for advanced
- **No Circular Dependencies:** Unidirectional flow (convenience → scaler → core/utils)

**Verdict:** ✅ Textbook modular design. Model for Phase 2 packages.

---

### 1.3 Pipeline Orchestration Pattern

**Pattern:** **Command Pattern + Dependency Graph**

```python
# Stage definitions with dependencies
stages = [
    {"name": "data_generation", "dependencies": []},
    {"name": "data_cleaning", "dependencies": ["data_generation"]},
    {"name": "feature_engineering", "dependencies": ["data_cleaning"]},
    {"name": "initial_labeling", "dependencies": ["feature_engineering"]},
    # ... etc
]

# Runner resolves dependencies and executes
for stage in stages_to_run:
    if dependencies_met(stage):
        result = stage.function()
        track_result(result)
```

**Strengths:**
- Linear dependency chain (no DAG complexity needed for Phase 1)
- Checkpointing after each stage (resume capability)
- Clear fail-fast behavior (required stages halt pipeline)
- Manifest tracking for artifacts

**Limitations for Phase 2:**
- No parallel execution support (single-threaded)
- No conditional branching (e.g., "train model A OR model B")
- No dynamic stage registration (all stages hardcoded)

**Verdict:** ✅ Perfect for Phase 1 linear pipeline. Needs enhancement for Phase 2 multi-model training.

---

## 2. Separation of Concerns Assessment

### 2.1 Configuration Management

**Location:** `src/config.py` (467 lines), `src/pipeline_config.py` (456 lines)

**Separation:**
```
config.py               → Global constants (SYMBOLS, BARRIER_PARAMS, PURGE_BARS)
pipeline_config.py      → Run-specific config (PipelineConfig dataclass)
```

**Strengths:**
- ✅ **Validation at boundaries:** `validate_config()` checks PURGE_BARS >= max(max_bars)
- ✅ **Symbol-specific parameters:** `get_barrier_params(symbol, horizon)` handles MES vs MGC
- ✅ **Immutable after validation:** Config validated in `__post_init__`
- ✅ **Serialization support:** `save_config()` / `load_config()` for reproducibility

**Issues:**
- ⚠️ **Two config modules:** `config.py` vs `pipeline_config.py` creates confusion
  - **Recommendation:** Merge into single `config/` package with `constants.py` and `pipeline.py`

**Verdict:** 8/10. Strong design, but needs consolidation.

---

### 2.2 Data Flow & Artifact Management

**Pattern:** **Manifest-based versioning** (`src/manifest.py`)

```python
manifest.add_artifact(
    name="features_MES",
    file_path=Path("data/features/MES_5m_features.parquet"),
    stage="feature_engineering",
    metadata={'feature_count': 50, 'row_count': 100000}
)
```

**Strengths:**
- ✅ **Checksum verification:** SHA256 hashing for data integrity
- ✅ **Stage provenance:** Track which stage produced which artifact
- ✅ **Metadata storage:** Captures row counts, feature counts, etc.
- ✅ **Run comparison:** `compare_runs()` for A/B testing

**Limitations for Phase 2:**
- No support for model artifacts (weights, hyperparameters)
- No lineage tracking (feature → model → prediction)
- No artifact expiration/cleanup policies

**Verdict:** ✅ Excellent foundation. Extend with `ModelArtifact` class for Phase 2.

---

### 2.3 Dependency Direction Analysis

**Dependency Graph (simplified):**
```
pipeline_cli.py
    ↓
pipeline/runner.py
    ↓
pipeline/stages/*.py (wrappers)
    ↓
stages/*.py (domain logic)
    ↓
config.py, manifest.py
    ↓
utils/ (shared utilities)
```

**Violations Found:** ❌ **1 Coupling Issue**
- `stage5_ga_optimize.py` imports `from stage4_labeling import triple_barrier_numba`
  - **Problem:** Direct import creates tight coupling between stages
  - **Fix:** Move `triple_barrier_numba` to `utils/labeling.py` for shared access

**Verdict:** 9/10. One minor coupling issue, otherwise clean.

---

## 3. Modularity Assessment

### 3.1 File Size Compliance (650 Line Limit)

**Files Exceeding 650 Lines:**
```
1729 lines  src/stages/feature_scaler_old.py    ❌ (archived, OK)
 988 lines  src/stages/generate_report.py       ❌ VIOLATION
 967 lines  src/stages/stage2_clean_old.py      ❌ (archived, OK)
 920 lines  src/stages/stage5_ga_optimize.py    ❌ VIOLATION
 900 lines  src/stages/stage8_validate.py       ❌ VIOLATION
 780 lines  src/pipeline_cli.py                 ❌ VIOLATION
 740 lines  src/stages/stage1_ingest.py         ❌ VIOLATION
```

**Active Violations:** 5 files (excluding _old.py archives)

**Refactoring Priorities:**
1. **stage5_ga_optimize.py (920 lines):** Split into `ga_optimize/` package
   - `ga_optimize/fitness.py` (fitness function)
   - `ga_optimize/evolution.py` (DEAP logic)
   - `ga_optimize/optimizer.py` (main class)

2. **generate_report.py (988 lines):** Split into `reporting/` package
   - `reporting/formatters.py` (markdown, JSON)
   - `reporting/aggregators.py` (summary stats)
   - `reporting/generator.py` (main class)

3. **stage8_validate.py (900 lines):** Split into `validation/` package
   - `validation/checks.py` (individual validators)
   - `validation/leakage.py` (leakage detection)
   - `validation/validator.py` (main class)

**Verdict:** 6/10. Needs refactoring for 5 files.

---

### 3.2 Package Structure Quality

**Well-Modularized Packages:**
```
✅ src/stages/feature_scaler/    (7 modules, max 546 lines)
✅ src/stages/stage2_clean/      (4 modules, max 589 lines)
✅ src/stages/features/          (11 modules, max 577 lines)
✅ src/pipeline/stages/          (10 modules, max 165 lines)
```

**Monolithic Files:**
```
❌ src/stages/stage1_ingest.py         (740 lines, should be package)
❌ src/stages/stage4_labeling.py       (506 lines, OK but near limit)
❌ src/stages/stage5_ga_optimize.py    (920 lines, needs package)
❌ src/stages/stage6_final_labels.py   (555 lines, OK but near limit)
❌ src/stages/stage7_splits.py         (432 lines, OK)
```

**Verdict:** 7/10. Good progress on refactoring, but 5 files need attention.

---

## 4. Phase 2 Readiness: Multi-Model Training Infrastructure

### 4.1 Missing Infrastructure

**Critical Gaps:**

#### A. Data Loading Layer
```python
# NEEDED: src/models/datasets/time_series.py
class TimeSeriesDataset(torch.utils.data.Dataset):
    """PyTorch dataset for time series prediction."""

    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        sequence_length: int,
        forecast_horizon: int
    ):
        # Sliding window logic for temporal sequences
        pass

    def __getitem__(self, idx):
        # Return (X_window, y_target, metadata)
        pass

# NEEDED: src/models/datasets/data_module.py
class FinancialDataModule(pl.LightningDataModule):
    """Lightning data module for train/val/test splits."""

    def setup(self, stage: str):
        # Load scaled splits from data/splits/scaled/
        pass

    def train_dataloader(self):
        return DataLoader(self.train_dataset, ...)
```

**Status:** ❌ **Not implemented**

---

#### B. Model Abstraction Layer
```python
# NEEDED: src/models/base/base_model.py
class BaseTimeSeriesModel(ABC):
    """Abstract base for all time series models."""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def configure_optimizers(self):
        pass

    def training_step(self, batch, batch_idx):
        # Shared training logic
        pass

# NEEDED: src/models/architectures/
#   - nhits.py (N-HiTS implementation)
#   - tft.py (Temporal Fusion Transformer)
#   - patchtst.py (PatchTST implementation)
#   - lstm.py (Baseline LSTM)
```

**Status:** ❌ **Not implemented**

---

#### C. Model Registry Pattern
```python
# NEEDED: src/models/registry.py
class ModelRegistry:
    """Registry for model architectures."""

    _models = {}

    @classmethod
    def register(cls, name: str):
        def decorator(model_class):
            cls._models[name] = model_class
            return model_class
        return decorator

    @classmethod
    def get(cls, name: str, **kwargs):
        return cls._models[name](**kwargs)

# Usage:
@ModelRegistry.register("nhits")
class NHiTS(BaseTimeSeriesModel):
    pass

@ModelRegistry.register("tft")
class TemporalFusionTransformer(BaseTimeSeriesModel):
    pass
```

**Status:** ❌ **Not implemented**

---

#### D. Training Orchestration
```python
# NEEDED: src/models/trainer/trainer.py
class ModelTrainer:
    """Orchestrates multi-model training."""

    def train_model(
        self,
        model_name: str,
        data_module: FinancialDataModule,
        config: ModelConfig
    ) -> TrainingResult:
        # 1. Instantiate model from registry
        # 2. Setup callbacks (checkpointing, early stopping)
        # 3. Train with PyTorch Lightning
        # 4. Evaluate on val set
        # 5. Return metrics + artifacts
        pass

# NEEDED: src/models/trainer/callbacks.py
#   - PredictionSaver (save predictions for ensemble)
#   - MetricsLogger (log to MLflow/W&B)
#   - ModelCheckpoint (save best model)
```

**Status:** ❌ **Not implemented**

---

#### E. Experiment Tracking
```python
# NEEDED: src/models/experiments/tracker.py
class ExperimentTracker:
    """Wrapper for MLflow/Weights & Biases."""

    def log_params(self, params: Dict):
        pass

    def log_metrics(self, metrics: Dict):
        pass

    def log_artifact(self, path: Path):
        pass

    def log_model(self, model, name: str):
        pass
```

**Status:** ❌ **Not implemented**

---

### 4.2 Recommended Phase 2 Architecture

```
src/models/
├── __init__.py
├── datasets/
│   ├── __init__.py
│   ├── time_series.py ........... TimeSeriesDataset
│   ├── data_module.py ........... FinancialDataModule (Lightning)
│   └── transforms.py ............ Data augmentation (if needed)
│
├── base/
│   ├── __init__.py
│   ├── base_model.py ............ BaseTimeSeriesModel (ABC)
│   └── base_config.py ........... ModelConfig (dataclass)
│
├── architectures/
│   ├── __init__.py
│   ├── nhits.py ................. N-HiTS implementation
│   ├── tft.py ................... Temporal Fusion Transformer
│   ├── patchtst.py .............. PatchTST
│   ├── lstm.py .................. Baseline LSTM
│   └── transformer.py ........... Vanilla Transformer
│
├── registry.py .................. ModelRegistry (factory pattern)
│
├── trainer/
│   ├── __init__.py
│   ├── trainer.py ............... ModelTrainer (orchestration)
│   ├── callbacks.py ............. Custom Lightning callbacks
│   └── metrics.py ............... Custom metrics (Sharpe, etc.)
│
├── experiments/
│   ├── __init__.py
│   ├── tracker.py ............... ExperimentTracker (MLflow wrapper)
│   └── config.py ................ Experiment configuration
│
└── ensemble/
    ├── __init__.py
    ├── stacking.py .............. Stacking ensemble
    ├── voting.py ................ Voting ensemble
    └── blending.py .............. Blending ensemble
```

---

## 5. Configuration Design Review

### 5.1 Current Design

**Two-tier configuration:**
```
config.py (global)          pipeline_config.py (run-specific)
    ↓                              ↓
SYMBOLS = ['MES', 'MGC']    run_id = "20251221_120000"
BARRIER_PARAMS = {...}      symbols = ['MES', 'MGC']
PURGE_BARS = 60             train_ratio = 0.70
```

**Issues:**
- ⚠️ **Duplication:** `SYMBOLS` defined in both files
- ⚠️ **Inconsistent access:** Sometimes `config.PURGE_BARS`, sometimes `config.purge_bars`
- ⚠️ **No environment support:** No dev/staging/prod configs

---

### 5.2 Recommended Refactoring

```
src/config/
├── __init__.py ............... Exports: get_config(), SYMBOLS, etc.
├── constants.py .............. Immutable constants (SYMBOLS, TICK_VALUES)
├── barriers.py ............... BARRIER_PARAMS, get_barrier_params()
├── pipeline.py ............... PipelineConfig (dataclass)
├── model.py .................. ModelConfig (for Phase 2)
└── environments/
    ├── base.py ............... BaseConfig (shared defaults)
    ├── dev.py ................ Development overrides
    ├── staging.py ............ Staging overrides
    └── prod.py ............... Production overrides
```

**Usage:**
```python
from config import get_config, SYMBOLS, get_barrier_params

config = get_config(env='prod')  # Returns PipelineConfig
barriers = get_barrier_params('MES', 5)  # Symbol-specific
```

**Verdict:** Current design is functional but needs consolidation for Phase 2.

---

## 6. Dependency Management & Coupling

### 6.1 Inter-Stage Coupling Analysis

**Tight Coupling Found:**
```python
# stage5_ga_optimize.py
from stage4_labeling import triple_barrier_numba  # ❌ Direct stage import
```

**Impact:**
- Cannot run stage5 without stage4 in PYTHONPATH
- Cannot reuse triple_barrier logic elsewhere
- Breaks stage isolation principle

**Fix:**
```python
# Refactor to:
from utils.labeling import triple_barrier_numba  # ✅ Shared utility

# OR create labeling package:
from stages.labeling import triple_barrier_numba  # ✅ Domain package
```

---

### 6.2 External Dependency Analysis

**Key Dependencies:**
```
pandas, numpy .............. Data manipulation
numba ...................... Performance (triple_barrier_numba)
deap ....................... Genetic algorithm (stage5)
pydantic ................... NOT USED (should consider for validation)
pyarrow .................... Parquet I/O
scikit-learn ............... Scalers (RobustScaler, StandardScaler)
```

**Missing for Phase 2:**
```
torch ...................... PyTorch (deep learning)
pytorch-lightning .......... Training framework
mlflow ..................... Experiment tracking
optuna ..................... Hyperparameter tuning (alternative to DEAP)
```

---

### 6.3 Import Analysis

**Clean Imports (✅):**
```python
# pipeline/stages/feature_engineering.py
from ..utils import StageResult, create_stage_result
from src.stages.stage3_features import FeatureEngineer
```

**Problematic Imports (⚠️):**
```python
# Multiple files use absolute imports from project root
from config import PURGE_BARS  # Works only if src/ in PYTHONPATH
from manifest import ArtifactManifest

# Better: Relative imports or package imports
from src.config import PURGE_BARS
from src.manifest import ArtifactManifest
```

**Verdict:** 8/10. Mostly clean, but inconsistent import styles.

---

## 7. Extensibility for Phase 2

### 7.1 Can We Add Multiple Model Families?

**Current Pipeline:**
```
Stage 1-8 → Scaled data → [MISSING: Model training]
```

**Required Extensions:**

#### Extension Point 1: Add Model Training Stage
```python
# pipeline/stage_registry.py
{
    "name": "train_models",
    "dependencies": ["feature_scaling"],
    "description": "Stage 9: Train multiple model families",
    "required": True,
    "stage_number": 9
}
```

#### Extension Point 2: Model Configuration
```python
# config/model.py
@dataclass
class ModelConfig:
    model_family: str  # 'nhits', 'tft', 'patchtst', etc.
    sequence_length: int = 60
    forecast_horizon: int = 5
    hidden_size: int = 128
    num_layers: int = 3
    dropout: float = 0.1
    learning_rate: float = 1e-3
```

#### Extension Point 3: Multi-Model Orchestration
```python
# pipeline/stages/model_training.py
def run_model_training(config, manifest):
    model_families = ['nhits', 'tft', 'patchtst', 'lstm']
    results = {}

    for model_name in model_families:
        model_config = ModelConfig(model_family=model_name)
        trainer = ModelTrainer(config=model_config)
        result = trainer.train(data_module)
        results[model_name] = result

    return create_stage_result(...)
```

**Verdict:** ✅ **Highly extensible.** Stage registry pattern supports new stages easily.

---

### 7.2 Can We Support Ensemble Methods?

**Current Output:**
```
data/splits/scaled/
├── MES_5m_train_scaled.parquet
├── MES_5m_val_scaled.parquet
└── MES_5m_test_scaled.parquet
```

**Ensemble Architecture:**
```
Stage 9: Train base models → Save predictions
    ↓
models/predictions/
├── nhits_train_preds.parquet
├── tft_train_preds.parquet
└── patchtst_train_preds.parquet
    ↓
Stage 10: Ensemble training → Meta-learner
    ↓
models/ensemble/
└── stacked_model.pkl
```

**Implementation Pattern:**
```python
# models/ensemble/stacking.py
class StackingEnsemble:
    def __init__(self, base_models: List[str], meta_learner: str):
        self.base_models = base_models
        self.meta_learner = meta_learner

    def fit(self, train_predictions: Dict[str, np.ndarray], y_train):
        # Train meta-learner on base model predictions
        pass

    def predict(self, val_predictions: Dict[str, np.ndarray]):
        # Generate ensemble predictions
        pass
```

**Verdict:** ✅ **Easily extensible** with additional pipeline stage.

---

## 8. Specific Architectural Improvements

### 8.1 High Priority (Do Before Phase 2)

#### 1. Consolidate Configuration (Priority: Critical)
```
ACTION: Merge config.py + pipeline_config.py → config/ package
IMPACT: Eliminates duplication, improves clarity
EFFORT: 4 hours
```

#### 2. Refactor Oversized Files (Priority: High)
```
ACTION: Split 5 files > 650 lines into packages
FILES: stage5_ga_optimize, generate_report, stage8_validate, pipeline_cli, stage1_ingest
IMPACT: Improved maintainability, testability
EFFORT: 16 hours (4h per file)
```

#### 3. Fix Stage Coupling (Priority: High)
```
ACTION: Move triple_barrier_numba to utils/labeling.py
IMPACT: Breaks stage4 → stage5 coupling
EFFORT: 1 hour
```

#### 4. Create Model Infrastructure (Priority: Critical)
```
ACTION: Build src/models/ package with:
  - datasets/time_series.py
  - base/base_model.py
  - registry.py
IMPACT: Enables Phase 2 model training
EFFORT: 24 hours
```

---

### 8.2 Medium Priority (Nice to Have)

#### 5. Add Pydantic Validation (Priority: Medium)
```python
# Current: Manual validation
def validate_config():
    if PURGE_BARS < max_max_bars:
        raise ValueError(...)

# Better: Pydantic models
from pydantic import BaseModel, Field, validator

class PipelineConfig(BaseModel):
    purge_bars: int = Field(ge=0)

    @validator('purge_bars')
    def validate_purge_bars(cls, v):
        # Automatic validation
        pass
```

#### 6. Implement Plugin Architecture (Priority: Medium)
```python
# Allow users to register custom stages
@register_stage(name="custom_feature", after="feature_engineering")
def run_custom_features(config, manifest):
    # Custom feature logic
    pass
```

#### 7. Add Parallel Execution (Priority: Medium)
```python
# Current: Sequential stage execution
# Future: Parallel execution for independent stages
@stage(dependencies=["data_cleaning"])
def feature_engineering_mes():
    pass

@stage(dependencies=["data_cleaning"])
def feature_engineering_mgc():
    pass

# Both can run in parallel
```

---

### 8.3 Low Priority (Future Work)

#### 8. Add Docker Support
```dockerfile
FROM python:3.10
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY src/ src/
CMD ["python", "-m", "src.pipeline_cli"]
```

#### 9. Implement CI/CD Pipeline
```yaml
# .github/workflows/pipeline.yml
name: Pipeline Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - run: pytest tests/
```

---

## 9. Technical Debt Assessment

### 9.1 Immediate Technical Debt

| Issue | Location | Impact | Effort |
|-------|----------|--------|--------|
| Oversized files (>650 lines) | 5 files | Maintainability | 16h |
| Stage coupling (stage4→stage5) | stage5_ga_optimize.py | Reusability | 1h |
| Config duplication | config.py, pipeline_config.py | Confusion | 4h |
| No model infrastructure | N/A | Blocks Phase 2 | 24h |
| Inconsistent imports | Multiple files | Clarity | 2h |

**Total Effort:** ~47 hours (1 week for 1 developer)

---

### 9.2 Architectural Debt

| Pattern | Current State | Ideal State | Priority |
|---------|---------------|-------------|----------|
| Configuration | Split across 2 files | Single config/ package | High |
| Model layer | Missing | Complete src/models/ | Critical |
| Validation | Manual checks | Pydantic models | Medium |
| Testing | 48/48 feature tests | 100% coverage | Low |
| Documentation | Good (MODULAR_ARCHITECTURE.md) | API docs (Sphinx) | Low |

---

## 10. Phase 2 Readiness Checklist

### 10.1 Infrastructure Requirements

- [ ] **TimeSeriesDataset** (PyTorch Dataset for sliding windows)
- [ ] **FinancialDataModule** (Lightning DataModule for splits)
- [ ] **BaseTimeSeriesModel** (Abstract base class)
- [ ] **ModelRegistry** (Factory pattern for model instantiation)
- [ ] **ModelTrainer** (Training orchestration)
- [ ] **ExperimentTracker** (MLflow/W&B integration)
- [ ] **Model configurations** (ModelConfig dataclass)
- [ ] **Callbacks** (Checkpointing, early stopping, metrics)
- [ ] **Ensemble infrastructure** (Stacking, voting, blending)

**Status:** 0/9 complete ❌

---

### 10.2 Architectural Cleanup

- [ ] Consolidate config.py + pipeline_config.py
- [ ] Refactor 5 oversized files into packages
- [ ] Fix stage4→stage5 coupling
- [ ] Standardize import style (relative vs absolute)
- [ ] Add Pydantic validation (optional but recommended)

**Status:** 0/5 complete ❌

---

## 11. Recommendations Summary

### 11.1 Before Starting Phase 2 (Week 1)

**Critical Path:**
1. **Create src/models/ package** (24h)
   - datasets/time_series.py (TimeSeriesDataset)
   - base/base_model.py (BaseTimeSeriesModel)
   - registry.py (ModelRegistry)

2. **Consolidate configuration** (4h)
   - Merge config.py + pipeline_config.py → config/ package

3. **Fix stage coupling** (1h)
   - Move triple_barrier_numba to utils/labeling.py

**Total:** ~29 hours (4 days for 1 developer)

---

### 11.2 During Phase 2 Development (Ongoing)

**Parallel Work:**
1. **Refactor oversized files** (16h)
   - stage5_ga_optimize.py → ga_optimize/ package
   - generate_report.py → reporting/ package
   - stage8_validate.py → validation/ package

2. **Add experiment tracking** (8h)
   - Integrate MLflow or Weights & Biases

3. **Implement ensemble infrastructure** (16h)
   - Stacking, voting, blending classes

**Total:** ~40 hours (1 week for 1 developer)

---

## 12. Final Verdict

### Phase 1 Architecture: 8.5/10 (Production-Ready)

**Strengths:**
- Excellent modular design (feature_scaler, stage2_clean packages)
- Strong separation of concerns (orchestration vs domain logic)
- Clear dependency direction (no circular dependencies)
- Comprehensive artifact tracking (manifest system)
- Good test coverage (48/48 feature scaler tests passing)

**Weaknesses:**
- 5 files exceed 650-line limit
- Configuration split across 2 files
- Minor stage coupling (stage4→stage5)
- No model infrastructure for Phase 2

---

### Phase 2 Readiness: 4/10 (Significant Work Needed)

**Missing Critical Infrastructure:**
- PyTorch Dataset/DataModule layer
- Model abstraction and registry
- Training orchestration
- Experiment tracking

**Estimated Effort:** ~70 hours (2 weeks for 1 developer)

---

## 13. Action Plan

### Week 1: Foundation (Critical Path)
- [ ] Day 1-2: Create src/models/datasets/ package
- [ ] Day 3: Implement ModelRegistry and BaseTimeSeriesModel
- [ ] Day 4: Consolidate configuration into config/ package
- [ ] Day 5: Fix stage coupling + code review

### Week 2: Model Implementations
- [ ] Day 1-2: Implement N-HiTS architecture
- [ ] Day 3: Implement baseline LSTM
- [ ] Day 4-5: Create ModelTrainer orchestration

### Week 3: Integration & Testing
- [ ] Day 1-2: Integrate experiment tracking (MLflow)
- [ ] Day 3: Add training pipeline stage
- [ ] Day 4-5: End-to-end testing + documentation

**Total Timeline:** 3 weeks to production-ready Phase 2 architecture.

---

## Appendix A: File Size Summary

```
VIOLATIONS (>650 lines):
  1729  feature_scaler_old.py (archived, exempt)
   988  generate_report.py (NEEDS REFACTOR)
   967  stage2_clean_old.py (archived, exempt)
   920  stage5_ga_optimize.py (NEEDS REFACTOR)
   900  stage8_validate.py (NEEDS REFACTOR)
   780  pipeline_cli.py (NEEDS REFACTOR)
   740  stage1_ingest.py (NEEDS REFACTOR)

COMPLIANT (<650 lines):
   589  stage2_clean/cleaner.py ✅
   577  features/engineer.py ✅
   546  feature_scaler/scaler.py ✅
   506  stage4_labeling.py ✅
   467  config.py ✅
   456  pipeline_config.py ✅
   432  stage7_splits.py ✅
```

---

## Appendix B: Dependency Graph

```
Orchestration Layer:
  pipeline_cli.py → pipeline/runner.py → pipeline/stages/*.py

Domain Layer:
  stages/stage2_clean/ (package)
  stages/feature_scaler/ (package)
  stages/features/ (package)
  stages/stage1_ingest.py
  stages/stage4_labeling.py
  stages/stage5_ga_optimize.py → stages/stage4_labeling.py (COUPLING)
  stages/stage6_final_labels.py
  stages/stage7_splits.py
  stages/stage8_validate.py

Configuration Layer:
  config.py
  pipeline_config.py
  manifest.py

Utilities Layer:
  utils/ (shared utilities)
  pipeline/utils.py (StageResult, StageStatus)
```

---

**End of Architecture Review**
