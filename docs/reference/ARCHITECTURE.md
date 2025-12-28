# ML Model Factory - Architecture Reference

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           ML MODEL FACTORY FOR OHLCV TIME SERIES                     │
│                          Single-Contract Plugin Architecture                         │
│                                                                                      │
│  Raw Data ──► Phase 1 ──► Phase 2 ──► Phase 3 ──► Phase 4 ──► Phase 5              │
│              (Data)      (Models)    (CV/OOF)   (Ensemble)  (Backtest)              │
│                                                                                      │
│  Status:     COMPLETE    COMPLETE    COMPLETE   COMPLETE    PLANNED                 │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

## Single-Contract Architecture

**Critical Design Decision: Each contract trains in complete isolation.**

### Key Principles

1. **One contract at a time** - Pipeline processes exactly one futures contract per run
2. **Complete isolation** - No cross-symbol features, labels, or correlation
3. **Symbol configurability** - Easy switching via configuration
4. **Separate model per symbol** - MES model ≠ MGC model

### Symbol Configuration

```python
# Specify contract to train
SYMBOL = "MES"  # or "MGC", "SI", "GC", etc.

# Data path resolution
data/raw/{symbol}_1m.parquet          # Input
data/splits/scaled/                   # Processed data (single symbol)
experiments/runs/{run_id}/            # Trained models (single symbol)
```

**Switching contracts:**
```bash
./pipeline run --symbols MES    # Train on MES
./pipeline run --symbols MGC    # Train on MGC (separate run)
```

**Multi-symbol processing blocked** by default (`allow_batch_symbols=False`).

---

## High-Level Data Flow

```
┌──────────────────┐
│   RAW OHLCV      │  data/raw/{SYMBOL}_1m.parquet
│   1-Minute Bars  │  - datetime, open, high, low, close, volume
└────────┬─────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                              PHASE 1: DATA PIPELINE                               │
│  ┌─────────┐  ┌─────────┐  ┌──────────┐  ┌────────┐  ┌────────┐  ┌─────────┐    │
│  │ INGEST  │─►│  CLEAN  │─►│ FEATURES │─►│ LABELS │─►│ SPLITS │─►│ SCALING │    │
│  │validate │  │resample │  │  150+    │  │triple- │  │purge/  │  │robust   │    │
│  │ OHLCV   │  │1m → 5m  │  │indicators│  │barrier │  │embargo │  │scaler   │    │
│  └─────────┘  └─────────┘  └──────────┘  └────────┘  └────────┘  └─────────┘    │
└──────────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────┐
│ SCALED DATASETS  │  data/splits/scaled/
│ train/val/test   │  - 221 features, labels, sample weights
└────────┬─────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌─────────┐ ┌─────────────────────────────────────────────────────────────────────┐
│ PHASE 2 │ │                      PHASE 3: CROSS-VALIDATION                      │
│ TRAINING│ │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐    │
│         │ │  │ PURGED   │─►│ FEATURE  │─►│ OPTUNA   │─►│ OOF PREDICTIONS  │    │
│ Single  │ │  │ K-FOLD   │  │SELECTION │  │ TUNING   │  │ Stacking Dataset │    │
│ Model   │ │  │5 or 3    │  │MDA/MDI   │  │50 trials │  │ per model        │    │
│ Training│ │  └──────────┘  └──────────┘  └──────────┘  └──────────────────┘    │
└─────────┘ └─────────────────────────────────────────────────────────────────────┘
    │                │
    ▼                ▼
┌─────────┐   ┌───────────────┐
│ TRAINED │   │ OOF + STACKING│  data/stacking/
│ MODELS  │   │   DATASETS    │  - stacking_dataset_h{N}.parquet
└─────────┘   └───────┬───────┘
                      │
                      ▼
              ┌───────────────┐
              │   PHASE 4     │
              │   ENSEMBLE    │  Meta-learner on OOF predictions
              └───────┬───────┘
                      │
                      ▼
              ┌───────────────┐
              │   PHASE 5     │
              │   BACKTEST    │  Sharpe, drawdown, win rate (PLANNED)
              └───────────────┘
```

---

## Phase 1: Data Pipeline Architecture

### Stage Breakdown (11 Stages)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           PHASE 1 STAGE BREAKDOWN                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  STAGE 1: INGEST                    STAGE 2: CLEAN                              │
│  ┌────────────────────┐             ┌────────────────────┐                      │
│  │ DataIngestor       │             │ DataCleaner        │                      │
│  │ • Validate OHLCV   │────────────►│ • Resample 1m→5m   │                      │
│  │ • Fix violations   │             │ • Handle gaps      │                      │
│  │ • Standardize TZ   │             │ • Remove outliers  │                      │
│  └────────────────────┘             └─────────┬──────────┘                      │
│                                               │                                  │
│                                               ▼                                  │
│  STAGE 3: FEATURES                  STAGE 3.5: MTF                              │
│  ┌────────────────────┐             ┌────────────────────┐                      │
│  │ FeatureEngineer    │             │ MTFFeatureGenerator│                      │
│  │ • Price (3)        │◄───────────►│ • 15min, 30min     │                      │
│  │ • Moving Avg (15+) │             │ • 1h, 4h, daily    │                      │
│  │ • Momentum (20+)   │             │ • No lookahead     │                      │
│  │ • Volatility (15+) │             └────────────────────┘                      │
│  │ • Volume (8+)      │                                                         │
│  │ • Trend (5+)       │                                                         │
│  │ • Temporal (10+)   │                                                         │
│  │ • Regime (6+)      │                                                         │
│  │ • Microstructure   │                                                         │
│  │ • Wavelets (15+)   │                                                         │
│  └─────────┬──────────┘                                                         │
│            │                                                                     │
│            ▼                                                                     │
│  STAGE 4: LABELING                  STAGE 5: GA OPTIMIZE                        │
│  ┌────────────────────┐             ┌────────────────────┐                      │
│  │ Triple-Barrier     │             │ Optuna Optimizer   │                      │
│  │ • k_up barrier     │────────────►│ • Optimize k_up    │                      │
│  │ • k_down barrier   │             │ • Optimize k_down  │                      │
│  │ • max_bars timeout │             │ • Sharpe fitness   │                      │
│  │ • Horizons: 5,10,  │             └─────────┬──────────┘                      │
│  │   15,20 bars       │                       │                                  │
│  └─────────┬──────────┘                       │                                  │
│            │◄─────────────────────────────────┘                                  │
│            ▼                                                                     │
│  STAGE 6: FINAL LABELS              STAGE 7: SPLITS                             │
│  ┌────────────────────┐             ┌────────────────────┐                      │
│  │ Quality Scoring    │             │ Time-Series Split  │                      │
│  │ • Sample weights   │────────────►│ • Train: 70%       │                      │
│  │   (0.5x - 1.5x)    │             │ • Val: 15%         │                      │
│  │ • MAE/MFE ratios   │             │ • Test: 15%        │                      │
│  │ • Pain-to-gain     │             │ • Purge: 60 bars   │                      │
│  └────────────────────┘             │ • Embargo: 1440    │                      │
│                                     └─────────┬──────────┘                      │
│                                               │                                  │
│                                               ▼                                  │
│  STAGE 7.5: SCALING                 STAGE 8: VALIDATION                         │
│  ┌────────────────────┐             ┌────────────────────┐                      │
│  │ RobustScaler       │             │ Quality Checks     │                      │
│  │ • Fit on TRAIN     │────────────►│ • No leakage       │                      │
│  │ • Transform all    │             │ • Feature quality  │                      │
│  │ • Clip outliers    │             │ • Label integrity  │                      │
│  │   [-5σ, +5σ]       │             └────────────────────┘                      │
│  └────────────────────┘                                                         │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Phase 1 Output Files

```
data/splits/scaled/
├── train_scaled.parquet     # 24,711 samples × 221 features
├── val_scaled.parquet       # 3,808 samples
├── test_scaled.parquet      # 3,869 samples
├── feature_scaler.pkl       # RobustScaler fitted on train only
└── split_config.json        # Split configuration

runs/{run_id}/artifacts/
├── manifest.json            # Pipeline artifacts manifest
├── pipeline_state.json      # Stage completion state
├── feature_set_manifest.json
└── dataset_manifest.json
```

### Leakage Prevention Mechanisms

1. **Purge/Embargo at Splits**
   - Purge: 60 bars (3× max horizon = 3×20)
   - Embargo: 1440 bars (~5 days at 5min)

2. **Invalid Labels**
   - Last `max_bars` samples have NaN labels
   - Prevents using future data for labeling

3. **Train-Only Scaling**
   - RobustScaler fit on training set only
   - Transform applied to val/test

4. **MTF Shift**
   - Multi-timeframe features shift by one HTF bar
   - Ensures no lookahead from higher timeframes

---

## Phase 2: Model Factory Architecture

### Plugin-Based Model Registry

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           MODEL FACTORY ARCHITECTURE                             │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│                        ┌─────────────────────────────┐                          │
│                        │      MODEL REGISTRY         │                          │
│                        │   @register decorator       │                          │
│                        │   Plugin-based discovery    │                          │
│                        │   12 Models Registered      │                          │
│                        └─────────────┬───────────────┘                          │
│                                      │                                           │
│            ┌─────────────────────────┼─────────────────────────┐                │
│            │                         │                         │                │
│            ▼                         ▼                         ▼                │
│   ┌────────────────┐       ┌────────────────┐       ┌────────────────┐         │
│   │    BOOSTING    │       │     NEURAL     │       │   CLASSICAL    │         │
│   ├────────────────┤       ├────────────────┤       ├────────────────┤         │
│   │ • XGBoost      │       │ • LSTM         │       │ • RandomForest │         │
│   │ • LightGBM     │       │ • GRU          │       │ • Logistic     │         │
│   │ • CatBoost     │       │ • TCN          │       │ • SVM          │         │
│   ├────────────────┤       ├────────────────┤       └────────────────┘         │
│   │ GPU: optional  │       │ GPU: required  │                                   │
│   │ Scaling: NO    │       │ Scaling: YES   │       ┌────────────────┐         │
│   │ Sequences: NO  │       │ Sequences: YES │       │   ENSEMBLE     │         │
│   └────────────────┘       └────────────────┘       ├────────────────┤         │
│                                                      │ • Voting       │         │
│                                                      │ • Stacking     │         │
│                                                      │ • Blending     │         │
│                                                      └────────────────┘         │
│                                                                                  │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                          BASE MODEL INTERFACE                            │   │
│   ├─────────────────────────────────────────────────────────────────────────┤   │
│   │  class BaseModel(ABC):                                                   │   │
│   │      @abstractmethod fit(X_train, y_train, X_val, y_val) → Metrics      │   │
│   │      @abstractmethod predict(X) → PredictionOutput                      │   │
│   │      @abstractmethod save(path) / load(path)                            │   │
│   │      get_feature_importance() → Dict[str, float]                        │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Model Family Specifications

| Model | Family | GPU | Scaling | Sequences | Training Time (H20) |
|-------|--------|-----|---------|-----------|---------------------|
| XGBoost | Boosting | Optional | NO | NO | 1-3 min |
| LightGBM | Boosting | Optional | NO | NO | 1-2 min |
| CatBoost | Boosting | Optional | NO | NO | 2-5 min |
| LSTM | Neural | Required | YES | YES (60) | 5-15 min |
| GRU | Neural | Required | YES | YES (60) | 5-12 min |
| TCN | Neural | Required | YES | YES (60) | 8-20 min |
| RandomForest | Classical | NO | NO | NO | 30-60 sec |
| Logistic | Classical | NO | YES | NO | 10-20 sec |
| SVM | Classical | NO | YES | NO | 2-5 min |
| Voting | Ensemble | Inherited | Inherited | Inherited | Sum of base |
| Stacking | Ensemble | Inherited | Inherited | Inherited | Sum + meta |
| Blending | Ensemble | Inherited | Inherited | Inherited | Sum + meta |

### Trainer Workflow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              TRAINER WORKFLOW                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. Load Data                                                               │
│     ├─ TimeSeriesDataContainer from data/splits/scaled/                     │
│     └─ Extract X_train, y_train, X_val, y_val                              │
│                                                                              │
│  2. Prepare Model-Specific Format                                           │
│     ├─ Neural: Create sequences (seq_len=60)                                │
│     ├─ Boosting: Use tabular arrays                                         │
│     └─ Classical: Use tabular arrays (scaled if needed)                     │
│                                                                              │
│  3. Train with Validation Monitoring                                        │
│     ├─ Early stopping on validation loss                                    │
│     ├─ Learning rate scheduling                                             │
│     ├─ Gradient clipping (neural)                                           │
│     └─ Mixed precision (GPU, bfloat16)                                      │
│                                                                              │
│  4. Save Artifacts                                                          │
│     ├─ experiments/runs/{run_id}/checkpoints/model.*                        │
│     ├─ experiments/runs/{run_id}/metrics/training_metrics.json              │
│     └─ experiments/runs/{run_id}/config/model_config.yaml                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### GPU Optimization (RTX 4070 Ti)

- **Mixed Precision:** bfloat16 (Ada Lovelace architecture)
- **Batch Size:** 256 (neural), auto (boosting)
- **VRAM:** 12GB available
- **Gradient Clipping:** 1.0
- **Memory Estimation:** Automatic batch size adjustment

---

## Phase 3: Cross-Validation Architecture

### Purged K-Fold with Embargo

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PURGED K-FOLD CROSS-VALIDATION                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Standard K-Fold (WRONG):                                                   │
│  |--Train--|--Test--|--Train--|  ← Future data leaks into training          │
│                                                                              │
│  Purged K-Fold (CORRECT):                                                   │
│  |--Train--|PURGE|--Test--|EMBARGO|--Train--|                               │
│             60 bars       1440 bars                                          │
│                                                                              │
│  • Purge: Remove samples whose labels depend on test set                    │
│  • Embargo: Break serial correlation after test set                         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### CV Pipeline Components

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        CV PIPELINE COMPONENTS                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. PurgedKFold                                                             │
│     ├─ Configurable n_splits (3 or 5)                                       │
│     ├─ Auto-scaled purge/embargo from horizon                               │
│     └─ Time-series aware fold generation                                    │
│                                                                              │
│  2. WalkForwardFeatureSelector                                              │
│     ├─ MDA (Mean Decrease Accuracy) importance                              │
│     ├─ MDI (Mean Decrease Impurity) importance                              │
│     ├─ Select top N features per fold                                       │
│     └─ Track feature stability across folds                                 │
│                                                                              │
│  3. OptunaHyperparameterTuner                                               │
│     ├─ Model-specific search spaces                                         │
│     ├─ 50-100 trials per model                                              │
│     ├─ Variance penalty: score = mean_F1 - 0.1 * std_F1                     │
│     └─ Best params per (model, horizon)                                     │
│                                                                              │
│  4. OOFGenerator                                                            │
│     ├─ Train on each fold's training data                                   │
│     ├─ Predict on each fold's validation data                               │
│     ├─ Concatenate predictions (every sample has OOF pred)                  │
│     └─ Output: stacking_dataset_h{N}.parquet                                │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Stacking Dataset Schema

```python
# data/stacking/stacking_dataset_h20.parquet
{
    "xgboost_prob_short": float,      # XGBoost P(short)
    "xgboost_prob_neutral": float,    # XGBoost P(neutral)
    "xgboost_prob_long": float,       # XGBoost P(long)
    "lightgbm_prob_short": float,     # LightGBM P(short)
    "lightgbm_prob_neutral": float,   # LightGBM P(neutral)
    "lightgbm_prob_long": float,      # LightGBM P(long)
    "lstm_prob_short": float,         # LSTM P(short)
    "lstm_prob_neutral": float,       # LSTM P(neutral)
    "lstm_prob_long": float,          # LSTM P(long)
    "models_agree": bool,             # All models predict same class
    "avg_confidence": float,          # Mean max probability
    "avg_entropy": float,             # Mean prediction entropy
    "y_true": int                     # Actual label (0=short, 1=neutral, 2=long)
}
```

---

## Phase 4: Ensemble Architecture

### Ensemble Methods (3 Implemented)

1. **Voting Ensemble**
   - Combine predictions via weighted/unweighted averaging
   - Fast inference (parallel prediction)
   - No additional training required

2. **Stacking Ensemble**
   - Train meta-learner on OOF predictions
   - Learns optimal model combination
   - Requires Phase 3 OOF datasets

3. **Blending Ensemble**
   - Train meta-learner on holdout predictions
   - Simpler than stacking (no CV required)
   - Less data efficient than stacking

### Recommended Configurations

| Use Case | Models | Method | Expected Performance |
|----------|--------|--------|---------------------|
| Low Latency | XGBoost + LightGBM + CatBoost | Voting | Sharpe 0.8-1.0 |
| Balanced | XGBoost + LightGBM + RF | Blending | Sharpe 0.9-1.1 |
| Neural Focus | LSTM + GRU + TCN | Stacking | Sharpe 0.7-1.0 |
| Maximum Accuracy | All 12 models | Stacking | Sharpe 1.0-1.3 |

---

## Module Boundaries and Responsibilities

### src/phase1/ - Data Pipeline

- **Responsibility:** Transform raw OHLCV into model-ready datasets
- **Input:** `data/raw/{symbol}_1m.parquet`
- **Output:** `data/splits/scaled/*.parquet`
- **Key Files:**
  - `pipeline_config.py` - Configuration dataclass
  - `stages/` - 11 pipeline stages
  - `config/` - Feature sets, barrier configs

### src/models/ - Model Factory

- **Responsibility:** Train and evaluate all model types
- **Input:** `data/splits/scaled/*.parquet`
- **Output:** `experiments/runs/{run_id}/`
- **Key Files:**
  - `registry.py` - Plugin system (12 models registered)
  - `base.py` - BaseModel interface
  - `trainer.py` - Unified training orchestration
  - `boosting/`, `neural/`, `classical/`, `ensemble/` - Model implementations

### src/cross_validation/ - CV System

- **Responsibility:** Time-series CV, hyperparameter tuning, OOF generation
- **Input:** `data/splits/scaled/*.parquet`
- **Output:** `data/stacking/*.parquet`
- **Key Files:**
  - `purged_kfold.py` - PurgedKFold, ModelAwareCV
  - `feature_selector.py` - Walk-forward selection
  - `oof_generator.py` - OOF predictions
  - `cv_runner.py` - CV orchestration

### src/cli/ - Command Interface

- **Responsibility:** CLI for pipeline and model operations
- **Entry Point:** `./pipeline` (Typer CLI)
- **Key Files:**
  - `run_commands.py` - run, rerun commands
  - `status_commands.py` - status, validate commands

---

## Data Contracts

### Phase 1 → Phase 2 Contract

**File:** `data/splits/scaled/train_scaled.parquet`

**Schema:**
```python
{
    # 221 features (150+ base + MTF + wavelets)
    "feature_0": float,
    ...
    "feature_220": float,

    # Labels for all horizons
    "label_h5": int,      # 0=short, 1=neutral, 2=long
    "label_h10": int,
    "label_h15": int,
    "label_h20": int,

    # Sample weights (quality-based)
    "sample_weight_h5": float,   # 0.5-1.5x
    "sample_weight_h10": float,
    "sample_weight_h15": float,
    "sample_weight_h20": float,

    # Metadata
    "datetime": datetime64,
    "symbol": str
}
```

### Phase 3 → Phase 4 Contract

**File:** `data/stacking/stacking_dataset_h{horizon}.parquet`

**Schema:** OOF predictions from all base models (see Stacking Dataset Schema above)

---

## Configuration System

### Model Configs (YAML)

**Location:** `config/models/{model_name}.yaml`

**Structure:**
```yaml
model:
  type: xgboost
  family: boosting

training:
  n_estimators: 300
  learning_rate: 0.05
  max_depth: 6

device:
  use_gpu: true
  mixed_precision: true

early_stopping:
  enabled: true
  patience: 20
  min_delta: 0.001
```

### Pipeline Config (Python)

**Location:** `src/phase1/pipeline_config.py`

**Key Parameters:**
```python
@dataclass
class PipelineConfig:
    symbol: str = "MES"
    label_horizons: List[int] = field(default_factory=lambda: [5, 10, 15, 20])
    train_pct: float = 0.70
    val_pct: float = 0.15
    test_pct: float = 0.15
    purge_bars: int = 60
    embargo_bars: int = 1440
    allow_batch_symbols: bool = False  # Single-contract enforcement
```

---

## Directory Structure

```
/home/user/Research/
│
├── src/
│   ├── phase1/                      # Data Pipeline
│   │   ├── stages/                  # 11 pipeline stages
│   │   ├── config/                  # Feature sets, barriers
│   │   └── pipeline_config.py       # PipelineConfig dataclass
│   │
│   ├── models/                      # Model Factory (12 models)
│   │   ├── registry.py              # Plugin system
│   │   ├── base.py                  # BaseModel interface
│   │   ├── trainer.py               # Training orchestration
│   │   ├── boosting/                # XGBoost, LightGBM, CatBoost
│   │   ├── neural/                  # LSTM, GRU, TCN
│   │   ├── classical/               # Random Forest, Logistic, SVM
│   │   └── ensemble/                # Voting, Stacking, Blending
│   │
│   ├── cross_validation/            # CV System
│   │   ├── purged_kfold.py          # Time-series CV
│   │   ├── feature_selector.py      # Walk-forward selection
│   │   ├── oof_generator.py         # OOF predictions
│   │   └── cv_runner.py             # CV orchestration
│   │
│   └── cli/                         # CLI interface
│       ├── run_commands.py
│       └── status_commands.py
│
├── scripts/
│   ├── train_model.py               # Single model training
│   └── run_cv.py                    # Cross-validation
│
├── config/
│   └── models/                      # 12 YAML configs
│
├── data/
│   ├── raw/                         # {symbol}_1m.parquet
│   ├── splits/scaled/               # Phase 1 output
│   └── stacking/                    # Phase 3 OOF output
│
├── experiments/
│   └── runs/{run_id}/               # Trained models
│
└── tests/                           # 1592 passing, 13 skipped
    ├── models/                      # Model tests
    ├── cross_validation/            # CV tests
    └── phase1/                      # Pipeline tests
```

---

## Key Design Patterns

### 1. Plugin Architecture (Model Registry)

```python
# Registration
@ModelRegistry.register("xgboost")
class XGBoostModel(BaseModel):
    def fit(self, X_train, y_train, X_val, y_val):
        # Training logic
        pass

# Discovery
available_models = ModelRegistry.list_all()  # ['xgboost', 'lstm', ...]
model = ModelRegistry.get("xgboost")
```

### 2. Factory Pattern (Model Creation)

```python
# Create model from config
config = load_model_config("xgboost")
model = ModelRegistry.get(config["model"]["type"])
trainer = ModelTrainer(model, config)
```

### 3. Stage-Based Pipeline

```python
# Each stage is self-contained
class FeatureEngineeringStage:
    def run(self, input_path: Path) -> Dict[str, Path]:
        # Load input
        df = load(input_path)

        # Transform
        df_features = self.engineer_features(df)

        # Save output
        output_path = self.save(df_features)

        return {"features": output_path}
```

### 4. Data Contract Enforcement

```python
# TimeSeriesDataContainer enforces schema
container = TimeSeriesDataContainer.from_parquet(
    train_path="data/splits/scaled/train_scaled.parquet"
)

# Automatic validation
assert container.X_train.shape[1] == 221  # 221 features
assert container.y_train.dtype == np.int64  # Integer labels
assert len(container.sample_weights_train) == len(container.y_train)
```

---

## Performance Targets

### Phase 1 (Data Pipeline)

- **Runtime:** 3-8 minutes (150k samples)
- **Memory:** < 4GB RAM
- **Output:** 221 features, 4 label horizons

### Phase 2 (Model Training)

| Model | Training Time | F1 Score | Sharpe |
|-------|--------------|----------|--------|
| Random Forest | 30-60 sec | 0.40-0.50 | 0.5-0.7 |
| XGBoost | 1-3 min | 0.50-0.65 | 0.8-1.2 |
| LightGBM | 1-2 min | 0.48-0.62 | 0.7-1.1 |
| LSTM | 5-15 min | 0.45-0.60 | 0.6-1.0 |

### Phase 3 (Cross-Validation)

- **5-Fold CV:** 5x training time per model
- **Optuna Tuning:** 50-100 trials = 50-100x training time
- **OOF Generation:** Same as CV runtime

### Phase 4 (Ensemble)

- **Voting:** Negligible additional time
- **Stacking:** Meta-learner training (~30 sec)
- **Expected Improvement:** +10-20% F1 over best single model

---

## Notes

- All diagrams are ASCII for portability
- File paths are absolute where critical
- Target line count: 400-500 lines (current: ~490)
- Comprehensive coverage of Phases 1-4
- Phase 5 (backtesting) marked as PLANNED
