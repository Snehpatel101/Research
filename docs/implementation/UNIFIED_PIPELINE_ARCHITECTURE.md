# Unified ML Pipeline Architecture

**Date:** 2026-01-16  
**Goal:** Single cohesive pipeline from raw data → deployed model

---

## Problem Statement

**Current state (FRAGMENTED):**
- 23+ scattered scripts (`train_model.py`, `train_ensemble.py`, `train_meta_labeling.py`, etc.)
- Multiple disconnected src directories (`phase1/`, `models/`, `training/`, `features/`, `feature_selection/`, etc.)
- No automatic handoff between phases (user runs phase1, then manually runs training)
- Multiple config formats (PipelineConfig, ExperimentConfig, TrainerConfig, CLI args)
- Advanced features (meta-labeling, regime-aware, walk-forward) in separate scripts

**Desired state (UNIFIED):**
```python
from src import MLPipeline

pipeline = MLPipeline(symbol="MES", horizons=[20], models=["xgboost", "lstm"])
pipeline.run()  # ONE method runs EVERYTHING
```

---

## Architecture Overview

### Core Components

```
src/
├── pipeline/
│   ├── unified.py          # MLPipeline - master orchestrator
│   ├── config.py           # MLConfig - unified configuration
│   ├── state.py            # PipelineState - state management
│   └── phases/
│       ├── data.py         # Stages 1-6: Data pipeline
│       ├── optimization.py # Stages 7-9: Optuna optimization
│       ├── training.py     # Stages 10-15: Training pipeline
│       └── deployment.py   # Stage 16: Bundling & deployment
├── models/                 # Model implementations (unchanged)
├── features/               # Feature engineering (unchanged)
└── cli/
    └── unified_cli.py      # Single 'ml' CLI with subcommands
```

### Complete 16-Stage Data Flow

```
MLPipeline(config)
  ↓
┌─────────────────────────────────────────────────────────────────┐
│                    DATA PREPARATION (Stages 1-6)                 │
├─────────────────────────────────────────────────────────────────┤
│  Stage 1: Ingestion - Load raw 1-min OHLCV                      │
│  Stage 2: Cleaning - Resample, gap handling, validation         │
│  Stage 3: Sessions - Trading hours filtering (RTH/ETH)          │
│  Stage 4: MTF Upscaling - 9 timeframes from 1-min base          │
│  Stage 5: Features - 162 indicators (12 families)               │
│  Stage 6: Regime - Market regime detection (vol + trend)        │
└────────────────────────┬────────────────────────────────────────┘
                         ↓
  [Checkpoint: data/features/{symbol}_features.parquet]
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│               OPTUNA OPTIMIZATION (Stages 7-9)                   │
├─────────────────────────────────────────────────────────────────┤
│  Stage 7: OPTUNA Label Optimization - 100 trials                │
│           (barrier params: upper_mult, lower_mult,              │
│            horizon, atr_period)                                 │
│                         ↓                                        │
│  Stage 8: OPTUNA Feature Selection - 100 trials                 │
│           (binary include/exclude per feature)                  │
│                         ↓                                        │
│  Stage 9: OPTUNA Feature Pruning - 50 trials                    │
│           (importance-based removal)                            │
└────────────────────────┬────────────────────────────────────────┘
                         ↓
  [Checkpoint: data/optimized/{symbol}_optimized.parquet]
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│               PREPROCESSING (Stages 10-12)                       │
├─────────────────────────────────────────────────────────────────┤
│  Stage 10: Splits - Train/val/test (70/15/15)                   │
│            + purge (60 bars) + embargo (1440 bars)              │
│                         ↓                                        │
│  Stage 11: Scaling - Train-only robust scaling                  │
│                         ↓                                        │
│  Stage 12: Adaptation - 2D/3D/4D tensor per model type          │
│            (TabularAdapter/SequenceAdapter/MultiStreamAdapter)  │
└────────────────────────┬────────────────────────────────────────┘
                         ↓
  [Checkpoint: data/splits/scaled/{symbol}_{split}.parquet]
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│               TRAINING (Stages 13-15)                            │
├─────────────────────────────────────────────────────────────────┤
│  Stage 13: OPTUNA Hyperparameter Optimization                   │
│            100 trials per model (23 models)                     │
│                         ↓                                        │
│  Stage 14: Training - PurgedKFold CV, OOF generation            │
│            (5-fold CV, out-of-fold predictions)                 │
│                         ↓                                        │
│  Stage 15: Stacking - OOF alignment, meta-learner               │
│            (heterogeneous base models → meta-learner)           │
└────────────────────────┬────────────────────────────────────────┘
                         ↓
  [Checkpoint: experiments/runs/{run_id}/models/]
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│               DEPLOYMENT (Stage 16)                              │
├─────────────────────────────────────────────────────────────────┤
│  Stage 16: Bundling - Model + Scaler + Graph → Artifact         │
│            (ModelBundle V1.1.0 with PreprocessingGraph)         │
└────────────────────────┬────────────────────────────────────────┘
                         ↓
  [Output: experiments/runs/{run_id}/bundles/{model}_bundle.pkl]
```

---

## Unified Configuration

### MLConfig Dataclass

Merges PipelineConfig + ExperimentConfig + TrainerConfig:

```python
@dataclass
class MLConfig:
    # ===== DATA CONFIGURATION =====
    symbol: str
    start_date: str | None = None
    end_date: str | None = None
    timeframe: str = "5min"  # Primary timeframe
    
    # ===== FEATURE CONFIGURATION =====
    feature_mode: str = "auto"  # auto, full, minimal, hft_only
    enable_wavelets: bool = True
    enable_microstructure: bool = True
    enable_mtf: bool = True
    mtf_timeframes: list[str] | None = None
    output_timeframes: list[str] | None = None  # For heterogeneous training
    
    # ===== LABELING CONFIGURATION =====
    horizons: list[int] = field(default_factory=lambda: [5, 10, 15, 20])
    labeling_method: str = "triple_barrier"
    k_up: float | None = None  # Symbol-specific if None
    k_down: float | None = None
    max_bars: int | None = None
    
    # ===== SPLIT CONFIGURATION =====
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    purge_bars: int = 60
    embargo_bars: int = 1440
    
    # ===== MODEL CONFIGURATION =====
    models: list[str] | list[ModelConfig]
    global_feature_optimization: bool = False
    global_hyperparam_optimization: bool = False
    
    # ===== ENSEMBLE CONFIGURATION =====
    build_ensemble: bool = False
    ensemble_method: str = "stacking"
    meta_learner: str = "ridge_meta"
    
    # ===== TRAINING MODE =====
    training_mode: str = "standard"  # standard, walk_forward, regime_aware, meta_labeling
    cross_validate: bool = True
    cv_splits: int = 5
    
    # ===== EVALUATION =====
    evaluation_methods: list[str] = field(default_factory=lambda: ["cv"])  # cv, walk_forward, cpcv_pbo
    
    # ===== OUTPUT =====
    output_dir: Path = field(default_factory=lambda: Path("experiments/runs"))
    data_dir: Path = field(default_factory=lambda: Path("data"))
    run_id: str | None = None  # Auto-generated if None

    # ===== OPTUNA OPTIMIZATION =====
    optimize_labels: bool = True
    label_trials: int = 100
    optimize_features: bool = True
    feature_selection_trials: int = 100
    feature_pruning_trials: int = 50
    optimize_hyperparameters: bool = True
    hyperparam_trials_per_model: int = 100
```

---

## Optuna Optimization Stages

The pipeline includes 4 dedicated Optuna optimization stages that systematically tune the ML pipeline for optimal performance.

### Optuna Trial Summary

| Stage | Optimization Target | Trials | Search Space |
|-------|---------------------|--------|--------------|
| **Stage 7** | Triple Barrier Labels | 100 | barrier params |
| **Stage 8** | Feature Selection | 100 | binary include/exclude |
| **Stage 9** | Feature Pruning | 50 | importance threshold |
| **Stage 13** | Hyperparameters | 100 per model | model-specific |

**Total Trials:** ~100 + 100 + 50 + (100 x N_models) = 250 + 100N trials

For 23 models: **2,550 total Optuna trials**

---

### Stage 7: Triple Barrier Label Optimization (OPTUNA)

**Configuration:** `config/optimization/label_optimization.yaml`
**Purpose:** Find optimal barrier parameters that maximize label quality and downstream model performance.

**Search Space (100 trials):**

| Parameter | Range | Description |
|-----------|-------|-------------|
| `upper_mult` | 1.0 - 4.0 | Upper barrier ATR multiplier |
| `lower_mult` | 1.0 - 4.0 | Lower barrier ATR multiplier |
| `horizon` | 5 - 60 | Maximum bars to hold |
| `atr_period` | 7 - 28 | ATR calculation period |

**Objective Function:**
```python
def label_objective(trial: optuna.Trial) -> float:
    """Maximize label quality via quick model validation."""
    upper_mult = trial.suggest_float("upper_mult", 1.0, 4.0)
    lower_mult = trial.suggest_float("lower_mult", 1.0, 4.0)
    horizon = trial.suggest_int("horizon", 5, 60)
    atr_period = trial.suggest_int("atr_period", 7, 28)

    # Generate labels with trial params
    labels = generate_triple_barrier_labels(
        df, upper_mult, lower_mult, horizon, atr_period
    )

    # Quick validation with fast model (e.g., LightGBM)
    score = quick_cv_score(X, labels, n_splits=3)

    return score  # Maximize F1 or balanced accuracy
```

**Output:**
- `optimal_barrier_params.json` - Best barrier configuration
- `label_optimization_history.csv` - All trial results
- Labeled dataset with optimized triple-barrier targets

---

### Stage 8: Feature Selection Optimization (OPTUNA)

**Configuration:** `config/optimization/feature_selection.yaml`
**Purpose:** Select optimal feature subset via binary include/exclude decisions.

**Search Space (100 trials):**

| Parameter | Type | Description |
|-----------|------|-------------|
| `feature_{i}` | bool | Include feature i (0 or 1) |
| 162 binary params | Categorical | One per feature |

**Strategy:**
1. Start with all 162 features
2. Each trial samples a binary mask over features
3. Train quick model on selected features
4. Optimize for validation F1 score

**Objective Function:**
```python
def feature_selection_objective(trial: optuna.Trial) -> float:
    """Maximize model performance with feature subset."""
    feature_mask = []
    for i, feature_name in enumerate(all_features):
        include = trial.suggest_categorical(f"feat_{i}", [True, False])
        feature_mask.append(include)

    # Select features based on mask
    X_selected = X[:, feature_mask]

    # Ensure minimum feature count
    if sum(feature_mask) < 10:
        return 0.0  # Penalize too few features

    # Quick CV evaluation
    score = quick_cv_score(X_selected, y, n_splits=3)

    return score
```

**Output:**
- `optimal_feature_mask.json` - Boolean mask for 162 features
- `selected_features.txt` - List of selected feature names
- Reduced feature matrix (typically 60-100 features)

---

### Stage 9: Feature Pruning Optimization (OPTUNA)

**Configuration:** `config/optimization/feature_pruning.yaml`
**Purpose:** Remove low-importance features based on model-derived importance scores.

**Search Space (50 trials):**

| Parameter | Range | Description |
|-----------|-------|-------------|
| `importance_threshold` | 0.001 - 0.1 | Minimum importance to keep |
| `top_k_features` | 20 - 100 | Maximum features to keep |
| `importance_method` | categorical | gain, split, shap |

**Strategy:**
1. Train model on features from Stage 8
2. Compute feature importance (gain, split, or SHAP)
3. Prune features below importance threshold
4. Optionally limit to top-K most important

**Objective Function:**
```python
def feature_pruning_objective(trial: optuna.Trial) -> float:
    """Maximize performance with importance-based pruning."""
    threshold = trial.suggest_float("importance_threshold", 0.001, 0.1, log=True)
    top_k = trial.suggest_int("top_k_features", 20, 100)
    method = trial.suggest_categorical("importance_method", ["gain", "split", "shap"])

    # Get feature importance from base model
    importance = compute_importance(model, X, method=method)

    # Prune by threshold and top-K
    keep_mask = (importance >= threshold)
    if keep_mask.sum() > top_k:
        keep_indices = np.argsort(importance)[-top_k:]
        keep_mask = np.zeros_like(keep_mask)
        keep_mask[keep_indices] = True

    X_pruned = X[:, keep_mask]
    score = quick_cv_score(X_pruned, y, n_splits=3)

    return score
```

**Output:**
- `pruned_feature_mask.json` - Final feature mask after pruning
- `feature_importance_ranking.csv` - Full importance scores
- Final feature matrix (typically 30-60 features)

---

### Stage 13: Hyperparameter Optimization (OPTUNA)

**Configuration:** `config/optimization/hyperparameter.yaml`
**Purpose:** Tune model-specific hyperparameters for each of 23 models.

**Trial Budget:** 100 trials per model = 2,300 trials total

**Search Spaces by Model Family:**

#### Boosting Models (XGBoost, LightGBM, CatBoost)
| Parameter | Range | Type |
|-----------|-------|------|
| `max_depth` | 3 - 12 | int |
| `learning_rate` | 0.001 - 0.3 | float (log) |
| `n_estimators` | 100 - 2000 | int |
| `min_child_weight` | 1 - 10 | int |
| `subsample` | 0.5 - 1.0 | float |
| `colsample_bytree` | 0.5 - 1.0 | float |
| `reg_alpha` | 1e-8 - 10 | float (log) |
| `reg_lambda` | 1e-8 - 10 | float (log) |

#### Neural Models (LSTM, GRU, TCN, Transformer)
| Parameter | Range | Type |
|-----------|-------|------|
| `hidden_size` | 32 - 512 | int |
| `num_layers` | 1 - 4 | int |
| `dropout` | 0.0 - 0.5 | float |
| `learning_rate` | 1e-5 - 1e-2 | float (log) |
| `batch_size` | 32 - 512 | int |
| `seq_len` | 10 - 100 | int |
| `weight_decay` | 1e-6 - 1e-2 | float (log) |

#### Advanced Transformers (PatchTST, iTransformer, TFT)
| Parameter | Range | Type |
|-----------|-------|------|
| `d_model` | 32 - 256 | int |
| `n_heads` | 2 - 8 | int |
| `n_layers` | 1 - 6 | int |
| `patch_len` | 4 - 32 | int |
| `stride` | 2 - 16 | int |
| `dropout` | 0.0 - 0.3 | float |
| `learning_rate` | 1e-5 - 1e-3 | float (log) |

#### Classical Models (RF, Logistic, SVM)
| Parameter | Range | Type |
|-----------|-------|------|
| `n_estimators` | 50 - 500 | int (RF) |
| `max_depth` | 3 - 20 | int (RF) |
| `C` | 1e-4 - 100 | float (log, SVM/LR) |
| `kernel` | rbf, linear, poly | categorical (SVM) |
| `penalty` | l1, l2, elasticnet | categorical (LR) |

**Objective Function:**
```python
def hyperparam_objective(trial: optuna.Trial, model_name: str) -> float:
    """Maximize CV performance for specific model."""
    # Sample hyperparameters based on model family
    params = sample_hyperparams(trial, model_name)

    # Build model with trial params
    model = build_model(model_name, params)

    # PurgedKFold cross-validation
    scores = []
    for train_idx, val_idx in purged_kfold.split(X, y):
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[val_idx])
        scores.append(f1_score(y[val_idx], pred, average='macro'))

    return np.mean(scores)
```

**Output per Model:**
- `{model}_best_params.json` - Optimal hyperparameters
- `{model}_optimization_history.csv` - All trial results
- `{model}_importance_plot.png` - Hyperparameter importance

---

### Optuna Configuration

```python
@dataclass
class OptunaConfig:
    """Configuration for all Optuna optimization stages."""

    # Stage 7: Label Optimization
    label_trials: int = 100
    label_sampler: str = "TPE"  # TPE, CMA-ES, Random
    label_pruner: str = "Hyperband"  # Hyperband, Median, None

    # Stage 8: Feature Selection
    feature_selection_trials: int = 100
    feature_selection_sampler: str = "TPE"
    min_features: int = 10
    max_features: int = 150

    # Stage 9: Feature Pruning
    feature_pruning_trials: int = 50
    importance_methods: list[str] = field(default_factory=lambda: ["gain", "shap"])

    # Stage 13: Hyperparameter Optimization
    hyperparam_trials_per_model: int = 100
    hyperparam_sampler: str = "TPE"
    hyperparam_pruner: str = "Hyperband"
    cv_folds: int = 5

    # General
    n_jobs: int = -1  # Parallel trials
    timeout_per_trial: int = 300  # 5 minutes max per trial
    storage: str | None = None  # SQLite/PostgreSQL for distributed
    study_name_prefix: str = "ml_factory"
```

---

## MLPipeline Class

### Interface

```python
class MLPipeline:
    def __init__(self, config: MLConfig | dict | str):
        """
        Initialize pipeline from:
        - MLConfig object
        - dict (backward compat)
        - YAML file path
        """

    # === STAGE GROUP EXECUTION ===
    def run(self) -> PipelineResult:
        """Run all 16 stages: data → optimization → training → bundling"""

    def run_data(self) -> DataPipelineResult:
        """Stages 1-6: Raw data → features + regime"""

    def run_optimization(self) -> OptimizationResult:
        """Stages 7-9: Optuna label/feature optimization"""

    def run_preprocessing(self) -> PreprocessingResult:
        """Stages 10-12: Splits → scaling → adaptation"""

    def run_training(self) -> TrainingResult:
        """Stages 13-15: Hyperparameter opt → training → stacking"""

    def run_bundling(self) -> BundlingResult:
        """Stage 16: Model + Scaler + Graph → Artifact"""
    
    # === STATE MANAGEMENT ===
    def save_state(self) -> None:
        """Save pipeline state for resumption"""
    
    def load_state(self, run_id: str) -> None:
        """Load previous pipeline state"""
    
    def resume(self, from_phase: str = "auto") -> PipelineResult:
        """Resume from checkpoint"""
    
    # === INSPECTION ===
    def get_state(self) -> PipelineState:
        """Get current pipeline state"""
    
    def get_results(self) -> dict:
        """Get all phase results"""
```

### Implementation Strategy

```python
class MLPipeline:
    def __init__(self, config: MLConfig | dict | str):
        self.config = self._normalize_config(config)
        self.state = PipelineState(run_id=self.config.run_id or self._generate_run_id())
        
        self._data_phase = DataPhase(self.config, self.state)
        self._training_phase = TrainingPhase(self.config, self.state)
        self._evaluation_phase = EvaluationPhase(self.config, self.state)
        self._deployment_phase = DeploymentPhase(self.config, self.state)
    
    def run(self) -> PipelineResult:
        """Run all phases sequentially with checkpointing."""
        results = {}
        
        if not self.state.is_phase_complete("data"):
            results["data"] = self.run_data()
        
        if not self.state.is_phase_complete("training"):
            results["training"] = self.run_training()
        
        if self.config.cross_validate or self.config.evaluation_methods:
            results["evaluation"] = self.run_evaluation()
        
        return PipelineResult(**results)
    
    def run_data(self) -> DataPipelineResult:
        """Delegate to existing phase1 pipeline."""
        from src.phase1.pipeline_config import PipelineConfig
        from src.phase1.runner import PipelineRunner
        
        pipeline_config = self._convert_to_pipeline_config()
        runner = PipelineRunner(pipeline_config)
        result = runner.run()
        
        self.state.mark_phase_complete("data", result)
        return DataPipelineResult(result)
    
    def run_training(self) -> TrainingResult:
        """Delegate to TrainingOrchestrator."""
        if not self.state.is_phase_complete("data"):
            raise RuntimeError("Must run data pipeline first")
        
        from src.training import TrainingOrchestrator, ExperimentConfig
        
        exp_config = self._convert_to_experiment_config()
        orchestrator = TrainingOrchestrator(exp_config)
        result = orchestrator.run()
        
        self.state.mark_phase_complete("training", result)
        return TrainingResult(result)
    
    def run_evaluation(self) -> EvaluationResult:
        """Run CV, walk-forward, CPCV-PBO based on config."""
        if not self.state.is_phase_complete("training"):
            raise RuntimeError("Must train models first")
        
        results = {}
        
        if "cv" in self.config.evaluation_methods:
            results["cv"] = self._run_cross_validation()
        
        if "walk_forward" in self.config.evaluation_methods:
            results["walk_forward"] = self._run_walk_forward()
        
        if "cpcv_pbo" in self.config.evaluation_methods:
            results["cpcv_pbo"] = self._run_cpcv_pbo()
        
        self.state.mark_phase_complete("evaluation", results)
        return EvaluationResult(results)
```

---

## Training Mode Integration

### Standard Training (Default)

```python
config = MLConfig(
    symbol="MES",
    horizons=[20],
    models=["xgboost", "lstm"],
    training_mode="standard",
)
pipeline = MLPipeline(config)
pipeline.run_training()
```

### Walk-Forward Training

```python
config = MLConfig(
    symbol="MES",
    horizons=[20],
    models=["xgboost"],
    training_mode="walk_forward",
    walk_forward_config={
        "window_size": 5000,
        "step_size": 1000,
        "min_train_size": 3000,
    },
)
pipeline = MLPipeline(config)
pipeline.run_training()
```

### Regime-Aware Training

```python
config = MLConfig(
    symbol="MES",
    horizons=[20],
    models=["xgboost"],
    training_mode="regime_aware",
    regime_config={
        "regimes": ["high_vol", "low_vol", "trending", "ranging"],
        "train_separate": True,
    },
)
pipeline = MLPipeline(config)
pipeline.run_training()
```

### Meta-Labeling

```python
config = MLConfig(
    symbol="MES",
    horizons=[20],
    models=["xgboost"],  # Primary model
    training_mode="meta_labeling",
    meta_labeling_config={
        "base_model": "xgboost",  # Generates side predictions
        "meta_model": "logistic",  # Sizes positions
    },
)
pipeline = MLPipeline(config)
pipeline.run_training()
```

---

## State Management

### PipelineState

```python
@dataclass
class PipelineState:
    run_id: str
    phases_completed: dict[str, bool] = field(default_factory=dict)
    phase_results: dict[str, Any] = field(default_factory=dict)
    checkpoints: dict[str, Path] = field(default_factory=dict)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: datetime | None = None
    
    def mark_phase_complete(self, phase: str, result: Any) -> None:
        self.phases_completed[phase] = True
        self.phase_results[phase] = result
        self._save_to_disk()
    
    def is_phase_complete(self, phase: str) -> bool:
        return self.phases_completed.get(phase, False)
    
    def get_checkpoint_path(self, phase: str) -> Path:
        return Path(f"experiments/runs/{self.run_id}/checkpoints/{phase}/")
    
    def _save_to_disk(self) -> None:
        path = Path(f"experiments/runs/{self.run_id}/pipeline_state.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2, default=str)
    
    @classmethod
    def load(cls, run_id: str) -> "PipelineState":
        path = Path(f"experiments/runs/{run_id}/pipeline_state.json")
        with open(path) as f:
            data = json.load(f)
        return cls(**data)
```

### Resumption Example

```python
# Original run crashes after data phase
pipeline = MLPipeline(config)
pipeline.run_data()  # ✅ Complete
# ... crash ...

# Resume from checkpoint
pipeline = MLPipeline.from_checkpoint(run_id="20260116_120000")
pipeline.resume()  # Skips data, continues from training
```

---

## Unified CLI

### Command Structure

```bash
ml data     # Run data pipeline (Phase 1-5)
ml train    # Train models (Phase 6)
ml evaluate # Run evaluation (Phase 7)
ml deploy   # Deploy model API (Phase 8)
ml run      # Run all phases
ml resume   # Resume from checkpoint
ml status   # Show pipeline status
ml clean    # Clean outputs
```

### Example Usage

```bash
# Full pipeline
ml run --symbol MES --horizons 20 --models xgboost,lstm --build-ensemble

# Data only
ml data --symbol MES --enable-wavelets --output-timeframes 9tf

# Training only (assumes data exists)
ml train --models xgboost,lstm --optimize-features --build-ensemble

# Walk-forward evaluation
ml evaluate --method walk_forward --window-size 5000

# Resume crashed run
ml resume --run-id 20260116_120000
```

---

## Model Family Compatibility

### Data Format Compatibility

| Model Family | Input Format | Adapter | Notes |
|--------------|--------------|---------|-------|
| **Tabular** (XGBoost, LightGBM, CatBoost, RF, Logistic, SVM) | 2D `(n_samples, n_features)` | `container.get_sklearn_arrays()` | Direct |
| **Neural** (LSTM, GRU, TCN, Transformer) | 3D `(n_samples, seq_len, n_features)` | `container.get_pytorch_sequences()` | Windowing |
| **Advanced** (PatchTST, iTransformer, TFT, N-BEATS) | 3D/4D multi-stream | Custom adapters | Multi-TF ingestion |

### Feature Compatibility Matrix

| Feature Type | Tabular | Neural | Advanced |
|--------------|---------|--------|----------|
| Momentum indicators | ✅ | ✅ | ❌ (raw OHLCV) |
| Volatility indicators | ✅ | ✅ | ❌ |
| Volume indicators | ✅ | ✅ | ❌ |
| Microstructure | ✅ | ✅ | ❌ |
| Wavelets | ✅ | ✅ | ❌ |
| MTF indicators | ✅ | ✅ | ❌ |
| Raw OHLCV | ✅ | ✅ | ✅ |

### Heterogeneous Ensemble Compatibility

**All models compatible for stacking** as long as:
1. **Same target labels** - All models predict same horizons
2. **Same train/val/test splits** - PurgedKFold splits identical
3. **OOF predictions** - Meta-learner uses out-of-fold predictions
4. **Same evaluation metrics** - F1, accuracy, Sharpe ratio

**Example heterogeneous stack:**
```python
config = MLConfig(
    models=[
        ModelConfig(name="xgboost", timeframe="15min", optimize_features=True),  # Tabular
        ModelConfig(name="lstm", timeframe="5min", optimize_features=True),       # Neural
        ModelConfig(name="patchtst", timeframe="1min"),                           # Advanced (raw)
    ],
    build_ensemble=True,
    meta_learner="ridge_meta",
)
```

**Compatibility guarantee:**
- XGBoost trains on optimized 15min features (~60 engineered)
- LSTM trains on optimized 5min features (~50 engineered)
- PatchTST trains on raw 1min OHLCV (5 raw features)
- Meta-learner stacks OOF predictions (all 3 → 1 ensemble prediction)
- **ALL from same 1-min canonical OHLCV source**

---

## Implementation Plan

### Implementation Phase 1: Data Pipeline (Stages 1-6)
1. Stage 1: Ingestion - Load raw 1-min OHLCV from parquet
2. Stage 2: Cleaning - Resample, gap handling, validation
3. Stage 3: Sessions - Trading hours filtering (RTH/ETH)
4. Stage 4: MTF Upscaling - 9 timeframes from 1-min base
5. Stage 5: Features - 162 indicators (12 families)
6. Stage 6: Regime - Market regime detection (vol + trend)

### Implementation Phase 2: Optuna Optimization (Stages 7-9)
7. Stage 7: Label Optimization - 100 trials (barrier params)
8. Stage 8: Feature Selection - 100 trials (binary include/exclude)
9. Stage 9: Feature Pruning - 50 trials (importance-based)

### Implementation Phase 3: Preprocessing (Stages 10-12)
10. Stage 10: Splits - Train/val/test (70/15/15) + purge/embargo
11. Stage 11: Scaling - Train-only robust scaling
12. Stage 12: Adaptation - 2D/3D/4D tensor per model type

### Implementation Phase 4: Training (Stages 13-15)
13. Stage 13: Hyperparameter Optimization - 100 trials per model
14. Stage 14: Training - PurgedKFold CV, OOF generation
15. Stage 15: Stacking - OOF alignment, meta-learner

### Implementation Phase 5: Deployment (Stage 16)
16. Stage 16: Bundling - Model + Scaler + Graph → Artifact

### Implementation Phase 6: Integration
17. Create `src/pipeline/unified.py` - MLPipeline class
18. Create `src/pipeline/config.py` - MLConfig + OptunaConfig
19. Create `src/cli/unified_cli.py` - Unified CLI
20. State management for 16-stage resume

**Total: 16 pipeline stages + 4 integration tasks**

---

## Optuna Trial Budget Summary

| Optimization Stage | Trials | Time Estimate |
|-------------------|--------|---------------|
| Stage 7: Label Optimization | 100 | ~10 min |
| Stage 8: Feature Selection | 100 | ~15 min |
| Stage 9: Feature Pruning | 50 | ~8 min |
| Stage 13: Hyperparameter (per model) | 100 | ~20-60 min |

**Per-Model Total:** ~250 trials + 100 hyperparam = 350 trials
**Full Pipeline (23 models):** 250 + (23 x 100) = **2,550 trials**

### Estimated Runtime (RTX 4090, 64GB RAM)

| Component | Time |
|-----------|------|
| Stages 1-6 (Data) | ~3 min |
| Stages 7-9 (Feature Opt) | ~33 min |
| Stage 10-12 (Preprocessing) | ~1 min |
| Stage 13 (Hyperparam, all models) | ~8-23 hours |
| Stage 14-15 (Training + Stacking) | ~1 hour |
| Stage 16 (Bundling) | ~5 min |

**Total (Full Pipeline):** ~10-25 hours (parallelizable across models)

---

## Success Criteria

**Complete when:**
1. Single `MLPipeline(config).run()` executes all 16 stages
2. All 4 Optuna optimization stages integrated (label, feature selection, pruning, hyperparameters)
3. All 23 models work together in heterogeneous ensembles
4. Walk-forward, regime-aware, meta-labeling training modes integrated
5. Single CLI with `ml run`, `ml data`, `ml optimize`, `ml train` subcommands
6. State management allows resume from any of 16 stages
7. Notebook uses unified interface
8. Zero LSP errors in core pipeline code

**NOT complete if:**
- User still needs to run stages separately
- Optuna optimization stages not automated
- Model families incompatible for stacking
- Multiple config formats required
- No state management/resumption

---

## 16-Stage Pipeline Quick Reference

| # | Stage | Description | Output |
|---|-------|-------------|--------|
| 1 | Ingestion | Load raw 1-min OHLCV | `{symbol}_1m.parquet` |
| 2 | Cleaning | Resample, gap handling | `{symbol}_1m_clean.parquet` |
| 3 | Sessions | Trading hours filter | Filtered DataFrame |
| 4 | MTF Upscaling | 9 timeframes | `{symbol}_{tf}.parquet` |
| 5 | Features | 162 indicators | `{symbol}_features.parquet` |
| 6 | Regime | Market regime detection | Regime labels |
| 7 | **OPTUNA: Labels** | 100 trials, barrier params | Optimized labels |
| 8 | **OPTUNA: Feature Selection** | 100 trials, binary | Feature mask |
| 9 | **OPTUNA: Feature Pruning** | 50 trials, importance | Pruned features |
| 10 | Splits | 70/15/15 + purge/embargo | Train/val/test |
| 11 | Scaling | Train-only robust | Scaled arrays |
| 12 | Adaptation | 2D/3D/4D tensors | Model-ready data |
| 13 | **OPTUNA: Hyperparameters** | 100 trials per model | Best params |
| 14 | Training | PurgedKFold CV, OOF | Trained models |
| 15 | Stacking | OOF alignment, meta-learner | Ensemble |
| 16 | Bundling | Model + Scaler + Graph | Inference bundle |
