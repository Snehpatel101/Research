# ML Model Factory for OHLCV Time Series

## Goal

Keep the codebase modular, readable, and easy to extend as we build a **model factory** that can train, evaluate, and compare ANY model type on OHLCV bar data.

This is not a single pipeline — it's a **factory** with a plugin architecture.

---

## OHLCV ML Modeling: Factory Pattern

We are building an **ML Model Factory** for OHLCV time series. The factory can train any model family (boosting, neural, transformers, classical ML, ensembles) using:

1. **Shared Data Contract** - All models consume identical preprocessed datasets
2. **Plugin-Based Model Registry** - Add new model types without rewriting pipelines
3. **Unified Evaluation Framework** - Compare models using identical metrics
4. **Ensemble Support Built-In** - Combine multiple models into meta-learners

### Factory Architecture Principles

```
Raw OHLCV → [ Data Pipeline ] → Standardized Datasets
                                       ↓
                            [ Model Registry Plugin System ]
                            ├── XGBoost Trainer
                            ├── LSTM Trainer
                            ├── Transformer Trainer
                            ├── Random Forest Trainer
                            └── Ensemble Meta-Learner
                                       ↓
                            [ Unified Evaluation Engine ]
                                       ↓
                          Trained Models + Performance Reports
```

### Core Contracts

**Data Contract (Phase 1 - Complete):**
- Clean/resample → features → labels → splits → scaling → datasets
- No lookahead bias (proper purging + embargo)
- Time-series aware train/val/test splits
- Quality-weighted samples

**Model Contract (Phase 2 - Complete):**
```python
class BaseModel(ABC):
    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series, config: dict) -> None:
        """Train the model on provided data."""
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions."""
        pass

    @abstractmethod
    def save(self, path: Path) -> None:
        """Persist trained model."""
        pass
```

**Evaluation Contract:**
- Same backtest assumptions for all models
- Identical metrics: Sharpe, win rate, max drawdown, transaction costs
- Regime-aware performance breakdown
- Quality-weighted evaluation

### Plugin Registration

Adding a new model type should be **trivial**:

```python
from src.models import ModelRegistry

@ModelRegistry.register("my_model")
class MyModel(BaseModel):
    def train(self, X, y, config):
        # Your training logic
        pass

    def predict(self, X):
        # Your prediction logic
        pass
```

Then use it:
```bash
./pipeline run --model-type my_model --symbols MES
```

### Ensemble Support

The factory supports both **single models** and **ensembles**:

```bash
# Train individual models
./pipeline run --model-type xgboost --symbols MES
./pipeline run --model-type lstm --symbols MES

# Train ensemble meta-learner
./pipeline run --model-type ensemble \
  --base-models xgboost,lstm \
  --meta-learner logistic
```

### Model Families (12 Models Implemented)

| Family | Models | Interface | Strengths | Status |
|--------|--------|-----------|-----------|--------|
| Boosting | XGBoost, LightGBM, CatBoost | `BoostingModel(BaseModel)` | Fast, interpretable, feature interactions | **Complete** |
| Neural | LSTM, GRU, TCN | `RNNModel(BaseModel)` | Temporal dependencies, sequential patterns | **Complete** |
| Classical | Random Forest, Logistic, SVM | `ClassicalModel(BaseModel)` | Robust baselines, interpretable | **Complete** |
| Ensemble | Voting, Stacking, Blending | `EnsembleModel(BaseModel)` | Combines diverse model strengths | **Complete** |

**All 12 models** implement the same `BaseModel` interface and consume the same standardized datasets from Phase 1.

**Registry:** Models register via `@ModelRegistry.register()` decorator for automatic discovery.

### Recommended Ensemble Configurations

| Ensemble Type | Models | Method | Use Case |
|---------------|--------|--------|----------|
| Boosting-Only | XGBoost + LightGBM + CatBoost | Voting | Low latency (< 5ms), production |
| Hybrid Fast | XGBoost + LightGBM + Random Forest | Voting/Blending | Balanced accuracy/speed |
| Neural Stack | LSTM + GRU + TCN | Stacking | Sequential pattern learning |
| Full Stack | All 12 models | Stacking with Logistic meta | Maximum accuracy, ensemble diversity |

**Implemented Ensemble Methods:**
- **Voting:** Combine predictions via weighted/unweighted averaging
- **Stacking:** Train meta-learner on base model out-of-fold predictions
- **Blending:** Train meta-learner on holdout predictions

See `docs/phases/PHASE_4.md` for detailed ensemble architecture and diversity metrics.

---

## Engineering Rules (Non-Negotiables)

### Architecture and Modularity
We do not build monoliths. Responsibilities must be split into small, composable modules with clear contracts and minimal coupling. Each module should do one thing well and expose a narrow, well-documented interface. Prefer dependency injection and explicit wiring over hidden globals or implicit side effects.

### File and Complexity Limits
No single file may exceed **650 lines**. If a module is growing, it's a signal that boundaries are wrong and responsibilities need to be refactored. Keep functions short, keep layers separated, and keep the cognitive load low.

### Fail Fast, Fail Hard
We would rather crash early than silently continue in an invalid state. Inputs are validated at the boundary. Assumptions are enforced with explicit checks. If something is wrong, we stop and surface a clear error message that points to the cause.

### Less Code is Better
Simpler implementations win. Prefer straightforward, boring solutions over clever abstractions. Avoid premature generalization. If a feature can be expressed with fewer moving parts, do that. Complexity must earn its place.

### Delete Legacy Code (If Unused, Remove It)
Legacy code is debt. If a file, function, or feature is not used, not referenced by any active code path, and not needed for the next planned milestone, delete it.

- Prefer deletion over commenting-out or leaving "dead" branches around.
- Remove unused imports, stale utilities, and orphaned tests/docs along with the code.
- If you're unsure whether something is needed, prove it's used (ripgrep call sites, run the feature, confirm tests). If you can't prove it, delete it.
- Git history is the archive — do not keep code "just in case."

### Concise Output and Minimal Documentation
Default to concise answers unless explicitly asked to expand.

Not every agent action needs a document. Write documentation only when it is needed for:
- An end-of-pass summary (what changed, what remains, and next steps)
- A decision/contract other work will depend on (schemas, interfaces, invariants)
- Investigation artifacts others must reuse (repro steps, evidence, links, key findings)

Otherwise, keep notes brief and inline (PR description, issue comment, short checklist).

### No Exception Swallowing
Do not paper over failures with try/except. We do not swallow errors or "recover" by guessing. Use explicit validation, explicit return types, and explicit preconditions. If a dependency can fail, make that failure visible in the function contract and test it. Exceptions are allowed to propagate naturally so failures are obvious and diagnosable.

### Clear Validation
Every boundary validates what it receives: configuration, CLI inputs, dataset schemas, feature matrices, labels, and model parameters. Validation errors must be actionable, specific, and consistent. Prefer schema-based validation and typed structures over ad hoc checks.

### Clear Tests
Every module ships with tests that prove the contract. Unit tests cover pure logic. Integration tests cover pipeline wiring and data flow. Regression tests lock down previously fixed issues. Tests should be deterministic, fast, and easy to run locally and in CI.

### Definition of Done
A change is complete only when:
- Implementation is modular
- Stays within file limits (650 lines)
- Validates inputs at boundaries
- Backed by tests that clearly demonstrate correctness

---

## Auto-Activated Agents

These agents trigger **automatically** based on context - no manual invocation needed:

| When You're Doing... | Agent Auto-Activates | What It Does |
|---------------------|---------------------|--------------|
| Building pipeline stages | `ml-engineer` | Creates DAG-based ML workflows |
| Writing feature code | `data-engineer` | Spark optimization, data pipelines |
| Designing labeling logic | `quant-analyst` | Trading strategies, risk metrics |
| Creating data classes | `python-pro` | Modern Python patterns, Pydantic |
| Adding tests | `tdd-orchestrator` | Red-green-refactor cycles |
| Debugging issues | `debugger` | Error investigation |
| Optimizing performance | `performance-engineer` | Profiling, caching |

---

## Build Commands

```bash
# Build new pipeline stage
/ml-pipeline-workflow "create stage for [description]"

# Build feature engineering
/data-engineering:spark-optimization "optimize feature calculation for [task]"

# Build labeling system
/quantitative-trading:quant-analyst "implement triple-barrier labeling"

# Build validation
/tdd-workflows:tdd-cycle "add validation for [component]"
```

---

## Sequential Build Flow

When building new functionality, agents chain automatically:

```
You say: "Build a new resampling stage"
         ↓
    ml-engineer activates (pipeline design)
         ↓
    data-engineer activates (implementation)
         ↓
    python-pro activates (code patterns)
         ↓
    tdd-orchestrator activates (tests)
```

---

## Context Auto-Save

Context saves automatically at:
- Stage completion → `runs/{run_id}/artifacts/pipeline_state.json`
- Checkpoint → `_save_state()` in PipelineRunner

Restore: `/context-restore --project research --mode full`

---

## Factory Data Pipeline (Phase 1 - Complete)

The data pipeline produces standardized datasets for all model types:

```
src/phase1/stages/
├── ingest/             → Load and validate raw data
├── clean/              → Resample 1min→5min, gap handling
├── features/           → 150+ indicators (momentum, wavelets, microstructure)
├── mtf/                → Multi-timeframe features
├── labeling/           → Triple-barrier initial labels
├── ga_optimize/        → Optuna parameter optimization
├── final_labels/       → Apply optimized parameters
├── splits/             → Train/val/test with purge/embargo
├── scaling/            → Train-only robust scaling
├── datasets/           → Build TimeSeriesDataContainer
├── validation/         → Feature correlation and quality checks
└── reporting/          → Generate completion reports
```

**Output:** Standardized datasets consumed by all model trainers

---

## Model Factory (Phase 2 - Complete)

Plugin-based model training system with **12 models across 4 families**:

```
src/models/
├── registry.py         → ModelRegistry plugin system (12 models registered)
├── base.py             → BaseModel interface
├── config.py           → TrainerConfig, YAML loading
├── trainer.py          → Unified training orchestration
├── device.py           → GPU detection, memory estimation
├── boosting/           → XGBoost, LightGBM, CatBoost (3 models)
├── neural/             → LSTM, GRU, TCN (3 models)
├── classical/          → Random Forest, Logistic, SVM (3 models)
└── ensemble/           → Voting, Stacking, Blending (3 models)
```

**Output:** Trained models + unified performance reports

**All 12 models available:** `xgboost`, `lightgbm`, `catboost`, `lstm`, `gru`, `tcn`, `random_forest`, `logistic`, `svm`, `voting`, `stacking`, `blending`

---

## Cross-Validation (Phase 3 - Complete)

Time-series aware cross-validation with purge/embargo:

```
src/cross_validation/
├── purged_kfold.py     → PurgedKFold with configurable purge/embargo
├── feature_selector.py → Walk-forward MDA/MDI feature selection
├── oof_generator.py    → Out-of-fold predictions for stacking
├── cv_runner.py        → CrossValidationRunner, Optuna tuning
└── param_spaces.py     → Hyperparameter search spaces
```

**Output:** CV results, OOF predictions, stacking datasets

---

## Quick Commands

```bash
# Run data pipeline (Phase 1)
./pipeline run --symbols MGC

# Train specific model (Phase 2)
python scripts/train_model.py --model xgboost --horizon 20
python scripts/train_model.py --model lstm --horizon 20 --seq-len 30
python scripts/train_model.py --model random_forest --horizon 20

# Train ensemble (Phase 4)
python scripts/train_model.py --model voting --base-models xgboost,lightgbm,lstm --horizon 20
python scripts/train_model.py --model stacking --base-models xgboost,lgbm,rf --horizon 20

# Run cross-validation (Phase 3)
python scripts/run_cv.py --models xgboost --horizons 20 --n-splits 5
python scripts/run_cv.py --models all --horizons 5,10,15,20 --tune

# List available models (should print 12)
python scripts/train_model.py --list-models
python -c "from src.models import ModelRegistry; print(len(ModelRegistry.list_all()))"
```

---

## Key Params

```python
SYMBOLS = ['MES', 'MGC']
LABEL_HORIZONS = [5, 10, 15, 20]  # All supported horizons
TRAIN/VAL/TEST = 70/15/15
# Purge/embargo are auto-scaled from max horizon:
# PURGE_BARS = max_horizon * 3 = 60 bars (prevents label leakage)
# EMBARGO_BARS = 1440 bars (~5 days at 5-min for serial correlation)
```

---

## Phase 1 Analysis Summary (2025-12-24)

### Overall Score: 8.5/10 (Production-Ready)

**Strengths:**
- Triple-barrier labeling with symbol-specific asymmetric barriers (MES: 1.5:1.0)
- Optuna-based parameter optimization with transaction cost penalties
- Proper purge (60) and embargo (1440) for leakage prevention
- Quality-based sample weighting (0.5x-1.5x)
- 150+ features including wavelets and microstructure
- Multi-timeframe analysis (5min to daily)
- TimeSeriesDataContainer for unified model training interface

**Recent Improvements:**
- Removed synthetic data generation - pipeline requires real data
- Added wavelet decomposition features
- Added microstructure features (bid-ask spread, order flow)
- Improved embargo to 1440 bars (5 days) for better serial correlation handling
- DataIngestor validates OHLCV data at pipeline entry

**Expected Performance:**
| Horizon | Sharpe | Win Rate | Max DD |
|---------|--------|----------|--------|
| H5 | 0.3-0.8 | 45-50% | 10-25% |
| H10 | 0.4-0.9 | 46-52% | 9-20% |
| H15 | 0.4-1.0 | 47-53% | 8-18% |
| H20 | 0.5-1.2 | 48-55% | 8-18% |
