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
│       ├── data.py         # Phase 1-5: Data pipeline
│       ├── training.py     # Phase 6: Model training
│       ├── evaluation.py   # Phase 7: CV/Walk-forward/CPCV-PBO
│       └── deployment.py   # Phase 8: Model serving
├── models/                 # Model implementations (unchanged)
├── features/               # Feature engineering (unchanged)
└── cli/
    └── unified_cli.py      # Single 'ml' CLI with subcommands
```

### Data Flow

```
MLPipeline(config)
  ↓
Phase 1-5: Data Pipeline (run_data)
  - Ingest raw 1-min OHLCV
  - Multi-timeframe upscaling (9 TFs)
  - Feature engineering (~180 features)
  - Triple-barrier labeling
  - Train/val/test splits + scaling
  ↓
  [Checkpoint: data/splits/scaled/{timeframe}/]
  ↓
Phase 6: Training (run_training)
  - Load TimeSeriesDataContainer
  - Per-model feature selection (strategies + Optuna)
  - Train base models (single-TF or MTF)
  - Build ensembles (heterogeneous stacking)
  ↓
  [Checkpoint: experiments/runs/{run_id}/models/]
  ↓
Phase 7: Evaluation (run_evaluation)
  - Cross-validation (PurgedKFold)
  - Walk-forward analysis
  - CPCV-PBO validation
  - Regime-aware performance
  ↓
  [Checkpoint: experiments/runs/{run_id}/evaluation/]
  ↓
Phase 8: Deployment (run_deployment)
  - Model serving API
  - Real-time inference
  - Monitoring dashboards
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
    
    # === PHASE EXECUTION ===
    def run(self) -> PipelineResult:
        """Run all phases: data → training → evaluation"""
    
    def run_data(self) -> DataPipelineResult:
        """Phase 1-5: Raw data → labeled features"""
    
    def run_training(self) -> TrainingResult:
        """Phase 6: Train models"""
    
    def run_evaluation(self) -> EvaluationResult:
        """Phase 7: CV/Walk-forward/CPCV-PBO"""
    
    def run_deployment(self) -> DeploymentResult:
        """Phase 8: Serve model API"""
    
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

### Phase 1: Core Architecture (2-3 hours)
1. Create `src/pipeline/unified.py` - MLPipeline class
2. Create `src/pipeline/config.py` - MLConfig dataclass
3. Create `src/pipeline/state.py` - PipelineState class
4. Wire to existing phase1 pipeline and TrainingOrchestrator

### Phase 2: Training Modes (2-3 hours)
5. Integrate walk-forward training
6. Integrate regime-aware training
7. Integrate meta-labeling
8. Add training mode dispatcher

### Phase 3: Evaluation (1-2 hours)
9. Integrate CV (existing)
10. Integrate walk-forward evaluation (existing)
11. Integrate CPCV-PBO (existing)

### Phase 4: CLI (1-2 hours)
12. Create `src/cli/unified_cli.py`
13. Consolidate script functionality
14. Add resume/status commands

### Phase 5: Testing & Documentation (2-3 hours)
15. Test end-to-end pipeline
16. Verify model family compatibility
17. Update notebook
18. Update CLAUDE.md

**Total: 8-13 hours**

---

## Success Criteria

✅ **Complete when:**
1. Single `MLPipeline(config).run()` executes full pipeline
2. All 23 models work together in heterogeneous ensembles
3. Walk-forward, regime-aware, meta-labeling integrated
4. Single CLI with `ml run`, `ml train`, `ml evaluate` subcommands
5. State management allows resume from any phase
6. Notebook uses unified interface
7. Zero LSP errors in core pipeline code

❌ **NOT complete if:**
- User still needs to run phase1 + training separately
- Advanced features require separate scripts
- Model families incompatible for stacking
- Multiple config formats required
- No state management/resumption
