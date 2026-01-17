# ML Factory: Unified Pipeline Architecture

**Version:** 2.0
**Purpose:** Single cohesive pipeline for Jupyter notebook consumption
**Design:** One entry point, centralized configuration

---

## 1. Core Design Philosophy

The ML Factory is a **SINGLE UNIFIED PIPELINE** - not a collection of separate tools. Everything flows through one entry point with one configuration object. The entire system is designed for use in a Jupyter notebook where users configure once and the pipeline handles everything automatically.

```
┌─────────────────────────────────────────────────────────────────┐
│                     JUPYTER NOTEBOOK                            │
│                                                                 │
│   config = PipelineConfig(...)   # ONE CONFIG OBJECT            │
│   pipeline = MLFactory(config)   # ONE ENTRY POINT              │
│   results = pipeline.run()       # ONE METHOD CALL              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     ML FACTORY PIPELINE                         │
│                                                                 │
│   Raw OHLCV ──► Features ──► Adapt ──► Train ──► Bundle        │
│                                                                 │
│   Everything automatic. No manual intervention.                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. The Single Entry Point

### MLFactory Class

```python
from ml_factory import MLFactory, PipelineConfig

# ONE configuration object controls EVERYTHING
config = PipelineConfig(
    # Data
    symbol="MES",
    data_path="./data/mes_1min.parquet",

    # Models to train
    models=["xgboost", "lightgbm", "lstm"],

    # Horizons
    horizons=[20],

    # Ensemble
    build_ensemble=True,
    meta_learner="ridge_meta",

    # Training mode
    training_mode="walk_forward",

    # Output
    output_dir="./experiments/exp_001",
)

# ONE entry point - MLFactory
pipeline = MLFactory(config)

# ONE method call - runs everything
results = pipeline.run()

# Results contain everything
print(results.metrics)           # All model metrics
print(results.bundle_path)       # Saved inference bundle
print(results.oof_predictions)   # OOF for analysis
```

---

## 3. Centralized Configuration

### PipelineConfig - The Single Source of Truth

```python
@dataclass
class PipelineConfig:
    """
    ONE configuration object that controls the ENTIRE pipeline.

    Set it once in your Jupyter notebook, and the pipeline
    handles everything else automatically.
    """

    # ═══════════════════════════════════════════════════════════
    # DATA CONFIGURATION
    # ═══════════════════════════════════════════════════════════
    symbol: str                          # Trading symbol (e.g., "MES")
    data_path: Path                      # Path to 1-min OHLCV parquet

    # ═══════════════════════════════════════════════════════════
    # MODEL CONFIGURATION
    # ═══════════════════════════════════════════════════════════
    models: list[str]                    # Models to train
    # Options: "xgboost", "lightgbm", "catboost", "random_forest",
    #          "logistic", "svm", "lstm", "gru", "tcn", "transformer",
    #          "patchtst", "itransformer", "tft", "nbeats",
    #          "inceptiontime", "resnet1d"

    horizons: list[int] = [20]           # Prediction horizons (bars)

    # ═══════════════════════════════════════════════════════════
    # TRAINING CONFIGURATION
    # ═══════════════════════════════════════════════════════════
    training_mode: str = "standard"
    # Options: "standard", "walk_forward", "regime_aware", "meta_labeling"

    cv_method: str = "purged_kfold"
    # Options: "purged_kfold", "cpcv", "walk_forward"

    n_splits: int = 5                    # CV folds
    purge_bars: int = 60                 # Gap before test
    embargo_bars: int = 1440             # Embargo after test

    # ═══════════════════════════════════════════════════════════
    # ENSEMBLE CONFIGURATION
    # ═══════════════════════════════════════════════════════════
    build_ensemble: bool = True          # Build stacking ensemble?
    meta_learner: str = "ridge_meta"
    # Options: "ridge_meta", "mlp_meta", "xgboost_meta", "calibrated_meta"

    # ═══════════════════════════════════════════════════════════
    # FEATURE CONFIGURATION
    # ═══════════════════════════════════════════════════════════
    feature_families: list[str] = "auto"
    # "auto" = pipeline selects per model type
    # Or specify: ["momentum", "volatility", "volume", "trend", ...]

    mtf_timeframes: list[str] = ["5min", "15min", "60min"]
    sequence_length: int = 60            # For neural models

    # ═══════════════════════════════════════════════════════════
    # LABELING CONFIGURATION
    # ═══════════════════════════════════════════════════════════
    labeling_method: str = "triple_barrier"
    # Options: "triple_barrier", "adaptive_barrier", "directional"

    optimize_labels: bool = True         # Optuna optimization

    # ═══════════════════════════════════════════════════════════
    # OUTPUT CONFIGURATION
    # ═══════════════════════════════════════════════════════════
    output_dir: Path                     # Where to save everything
    save_bundle: bool = True             # Save inference bundle
    save_oof: bool = True                # Save OOF predictions
```

---

## 4. What Happens When You Call `pipeline.run()`

```
pipeline.run()
     │
     ▼
┌────────────────────────────────────────────────────────────────┐
│ STAGE 1: DATA LOADING                                          │
│   Load raw OHLCV from config.data_path                         │
│   Validate schema, check for gaps                              │
└────────────────────────────────────────────────────────────────┘
     │
     ▼
┌────────────────────────────────────────────────────────────────┐
│ STAGE 2: MTF RESAMPLING                                        │
│   Resample 1-min to config.mtf_timeframes                      │
│   Create: 5min, 15min, 60min (or as configured)                │
└────────────────────────────────────────────────────────────────┘
     │
     ▼
┌────────────────────────────────────────────────────────────────┐
│ STAGE 3: FEATURE ENGINEERING                                   │
│   Compute 150+ indicators automatically                        │
│   Apply anti-lookahead shifts                                  │
│   Select features per model type (if "auto")                   │
└────────────────────────────────────────────────────────────────┘
     │
     ▼
┌────────────────────────────────────────────────────────────────┐
│ STAGE 4: LABELING                                              │
│   Apply config.labeling_method                                 │
│   Optuna optimization if config.optimize_labels=True           │
│   Generate labels for each horizon                             │
└────────────────────────────────────────────────────────────────┘
     │
     ▼
┌────────────────────────────────────────────────────────────────┐
│ STAGE 5: SPLITTING & SCALING                                   │
│   Train (70%) / Val (15%) / Test (15%)                         │
│   RobustScaler fit on train only                               │
│   Purge/embargo gaps applied                                   │
└────────────────────────────────────────────────────────────────┘
     │
     ▼
┌────────────────────────────────────────────────────────────────┐
│ STAGE 6: TRAINING (for each model in config.models)            │
│                                                                │
│   FOR model IN config.models:                                  │
│     1. Auto-select adapter (2D/3D/4D based on model)           │
│     2. Auto-select features (based on model family)            │
│     3. Train with config.cv_method                             │
│     4. Generate OOF predictions                                │
│     5. Compute metrics                                         │
└────────────────────────────────────────────────────────────────┘
     │
     ▼
┌────────────────────────────────────────────────────────────────┐
│ STAGE 7: ENSEMBLE (if config.build_ensemble=True)              │
│   Align OOF predictions (handles 2D/3D mixing)                 │
│   Build stacking features                                      │
│   Train config.meta_learner                                    │
└────────────────────────────────────────────────────────────────┘
     │
     ▼
┌────────────────────────────────────────────────────────────────┐
│ STAGE 8: BUNDLING                                              │
│   Create ModelBundle with:                                     │
│     - Trained models                                           │
│     - Fitted scaler                                            │
│     - PreprocessingGraph (for raw inference)                   │
│     - Metadata                                                 │
│   Save to config.output_dir                                    │
└────────────────────────────────────────────────────────────────┘
     │
     ▼
┌────────────────────────────────────────────────────────────────┐
│ RETURN: PipelineResults                                        │
│   .metrics          - All model metrics                        │
│   .oof_predictions  - OOF for analysis                         │
│   .bundle_path      - Path to saved bundle                     │
│   .training_time    - Total training time                      │
│   .config           - Config used (for reproducibility)        │
└────────────────────────────────────────────────────────────────┘
```

---

## 5. Jupyter Notebook Usage

### Basic Usage

```python
# Cell 1: Import and Configure
from ml_factory import MLFactory, PipelineConfig

config = PipelineConfig(
    symbol="MES",
    data_path="./data/mes_1min.parquet",
    models=["xgboost", "lightgbm", "lstm"],
    horizons=[20],
    build_ensemble=True,
    output_dir="./experiments/exp_001",
)

# Cell 2: Run Pipeline
pipeline = MLFactory(config)
results = pipeline.run()

# Cell 3: Analyze Results
print(f"Best model: {results.best_model}")
print(f"Best F1: {results.best_f1:.4f}")
results.plot_metrics()

# Cell 4: Use for Inference
bundle = results.load_bundle()
predictions = bundle.predict(new_data)
```

### Advanced Configuration

```python
# Walk-forward validation with regime-aware features
config = PipelineConfig(
    symbol="MES",
    data_path="./data/mes_1min.parquet",

    # Use all available models
    models=[
        "xgboost", "lightgbm", "catboost",      # Boosting
        "lstm", "gru", "tcn",                    # Neural
        "transformer", "tft",                    # Transformers
    ],

    horizons=[10, 20],                           # Multiple horizons

    # Walk-forward for production simulation
    training_mode="walk_forward",
    cv_method="walk_forward",

    # Full ensemble with XGBoost meta
    build_ensemble=True,
    meta_learner="xgboost_meta",

    # Include regime features
    feature_families=["momentum", "volatility", "volume", "regime"],

    output_dir="./experiments/production_run",
)

pipeline = MLFactory(config)
results = pipeline.run()
```

---

## 6. What the Pipeline Handles Automatically

| Task | Manual Work Required |
|------|---------------------|
| Feature engineering (150+ indicators) | None |
| MTF resampling (9 timeframes) | None |
| Data adaptation (2D/3D/4D) | None - auto per model |
| Feature selection per model | None - auto per family |
| CV with purge/embargo | None |
| OOF generation | None |
| OOF alignment (heterogeneous) | None |
| Scaler fitting (train only) | None |
| Anti-lookahead protection | None |
| Bundle creation | None |
| Inference graph serialization | None |

---

## 7. The Unified Pipeline Internals

```python
class MLFactory:
    """
    THE single entry point for the entire ML pipeline.

    This class orchestrates everything:
    - Data loading and validation
    - Feature engineering
    - Multi-timeframe processing
    - Labeling with optimization
    - Model training
    - Ensemble building
    - Bundle creation

    Users interact ONLY with this class.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._validate_config()

        # Internal components (users don't touch these)
        self._data_loader = DataLoader(config)
        self._feature_engine = FeatureEngine(config)
        self._labeler = Labeler(config)
        self._trainer = Trainer(config)
        self._ensemble_builder = EnsembleBuilder(config)
        self._bundler = Bundler(config)

    def run(self) -> PipelineResults:
        """
        Execute the ENTIRE pipeline with ONE method call.

        Returns:
            PipelineResults with metrics, OOF, bundle path, etc.
        """
        # Stage 1-2: Data
        data = self._data_loader.load()
        mtf_data = self._data_loader.resample_mtf(data)

        # Stage 3: Features
        features = self._feature_engine.compute_all(mtf_data)

        # Stage 4: Labels
        labels = self._labeler.create_labels(data)

        # Stage 5: Split & Scale
        splits = self._create_splits(features, labels)
        scaled = self._scale(splits)

        # Stage 6: Train all models
        model_results = {}
        for model_name in self.config.models:
            model_results[model_name] = self._trainer.train(
                model_name, scaled
            )

        # Stage 7: Ensemble
        ensemble_result = None
        if self.config.build_ensemble:
            ensemble_result = self._ensemble_builder.build(
                model_results, scaled
            )

        # Stage 8: Bundle
        bundle_path = self._bundler.create(
            model_results, ensemble_result
        )

        return PipelineResults(
            config=self.config,
            model_results=model_results,
            ensemble_result=ensemble_result,
            bundle_path=bundle_path,
        )
```

---

## 8. Model Selection Guide

```
QUICK START (pick based on your needs):

Fast iteration, interpretable:
    models=["xgboost", "lightgbm"]

Production ensemble:
    models=["xgboost", "lightgbm", "lstm"]
    build_ensemble=True

Maximum diversity:
    models=["xgboost", "lightgbm", "lstm", "transformer"]
    build_ensemble=True
    meta_learner="xgboost_meta"

GPU available, complex patterns:
    models=["lstm", "gru", "tcn", "transformer", "tft"]
    build_ensemble=True
```

---

## 9. Output Structure

```
config.output_dir/
│
├── config.json              # Saved config (reproducibility)
├── metrics.json             # All model metrics
├── training_log.txt         # Training logs
│
├── oof_predictions/         # OOF predictions for analysis
│   ├── xgboost_h20.parquet
│   ├── lightgbm_h20.parquet
│   └── lstm_h20.parquet
│
├── models/                  # Individual model artifacts
│   ├── xgboost_h20/
│   ├── lightgbm_h20/
│   └── lstm_h20/
│
└── bundle/                  # Final inference bundle
    ├── manifest.json
    ├── metadata.json
    ├── features.json
    ├── scaler.pkl
    ├── preprocessing_graph.json
    └── model/
```

---

## 10. Key Design Decisions

### Why One Entry Point?
- **Simplicity**: Users learn one class, one method
- **Consistency**: Pipeline stages always run in correct order
- **Reproducibility**: Config saved with results
- **No Errors**: Can't skip stages or misconfigure

### Why Centralized Config?
- **Single Source of Truth**: All settings in one place
- **Easy Experimentation**: Change config, re-run
- **Jupyter-Friendly**: Cell 1 = config, Cell 2 = run
- **Serializable**: Save/load configs for reproducibility

### Why Automatic Adaptation?
- **User Doesn't Need to Know**: XGBoost needs 2D, LSTM needs 3D
- **Fewer Errors**: Pipeline handles tensor shapes
- **Focus on Models**: Users pick models, not adapters

---

## 11. Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   ML FACTORY = ONE UNIFIED PIPELINE                             │
│                                                                 │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  PipelineConfig    = ONE config object                  │   │
│   │  MLFactory         = ONE entry point                    │   │
│   │  pipeline.run()    = ONE method call                    │   │
│   │  PipelineResults   = ONE results object                 │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│   Everything else is internal. Users don't touch it.            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Document Metadata

| Field | Value |
|-------|-------|
| Version | 2.0 |
| Created | 2026-01-17 |
| Purpose | Unified pipeline for Jupyter notebook |
| Design | One entry point, centralized config |
