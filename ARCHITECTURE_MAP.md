# ML Model Factory - Complete Architecture Map

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           ML MODEL FACTORY FOR OHLCV TIME SERIES                     │
│                                   12 Models Implemented                              │
│                                                                                      │
│  Raw Data ──► Phase 1 ──► Phase 2 ──► Phase 3 ──► Phase 4 ──► Phase 5              │
│              (Data)      (Models)    (CV/OOF)   (Ensemble)  (Backtest)              │
│                                                                                      │
│  Status:     COMPLETE    COMPLETE    COMPLETE   COMPLETE    PLANNED                 │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

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
              │   PHASE 4     │  (PLANNED)
              │   ENSEMBLE    │  Meta-learner on OOF predictions
              └───────┬───────┘
                      │
                      ▼
              ┌───────────────┐
              │   PHASE 5     │  (PLANNED)
              │   BACKTEST    │  Sharpe, drawdown, win rate
              └───────────────┘
```

---

## Phase 1: Data Pipeline (COMPLETE)

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

OUTPUT FILES:
├── data/splits/scaled/train_scaled.parquet  (24,711 samples × 221 features)
├── data/splits/scaled/val_scaled.parquet    (3,808 samples)
├── data/splits/scaled/test_scaled.parquet   (3,869 samples)
├── data/splits/scaled/feature_scaler.pkl
└── data/splits/split_config.json
```

---

## Phase 2: Model Factory (COMPLETE)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           MODEL FACTORY ARCHITECTURE                             │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│                        ┌─────────────────────────────┐                          │
│                        │      MODEL REGISTRY         │                          │
│                        │   @register decorator       │                          │
│                        │   Plugin-based discovery    │                          │
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
│   │ • CatBoost     │       │ • TCN          │       │ (planned)      │         │
│   ├────────────────┤       ├────────────────┤       └────────────────┘         │
│   │ GPU: optional  │       │ GPU: required  │                                   │
│   │ Scaling: NO    │       │ Scaling: YES   │                                   │
│   │ Sequences: NO  │       │ Sequences: YES │                                   │
│   └────────────────┘       └────────────────┘                                   │
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
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                              TRAINER                                     │   │
│   ├─────────────────────────────────────────────────────────────────────────┤   │
│   │  1. Load data from TimeSeriesDataContainer                              │   │
│   │  2. Prepare sequences (if neural) or arrays (if boosting)               │   │
│   │  3. Train model with validation monitoring                              │   │
│   │  4. Early stopping on validation loss                                   │   │
│   │  5. Save model + metrics + config                                       │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                         GPU OPTIMIZATION (4070 Ti)                       │   │
│   ├─────────────────────────────────────────────────────────────────────────┤   │
│   │  • Mixed precision: bfloat16 (Ada Lovelace)                             │   │
│   │  • Batch size: 256 (neural), auto (boosting)                            │   │
│   │  • Memory: 12GB VRAM                                                    │   │
│   │  • Gradient clipping: 1.0                                               │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘

MODEL SPECIFICATIONS (12 Total):
┌──────────────┬────────────┬────────────┬────────────┬────────────┐
│    Model     │   Family   │    GPU     │  Scaling   │ Sequences  │
├──────────────┼────────────┼────────────┼────────────┼────────────┤
│ XGBoost      │ Boosting   │ Optional   │ NO         │ NO         │
│ LightGBM     │ Boosting   │ Optional   │ NO         │ NO         │
│ CatBoost     │ Boosting   │ Optional   │ NO         │ NO         │
│ LSTM         │ Neural     │ Required   │ YES        │ YES (60)   │
│ GRU          │ Neural     │ Required   │ YES        │ YES (60)   │
│ TCN          │ Neural     │ Required   │ YES        │ YES (60)   │
│ RandomForest │ Classical  │ NO         │ NO         │ NO         │
│ Logistic     │ Classical  │ NO         │ YES        │ NO         │
│ SVM          │ Classical  │ NO         │ YES        │ NO         │
│ Voting       │ Ensemble   │ Inherited  │ Inherited  │ Inherited  │
│ Stacking     │ Ensemble   │ Inherited  │ Inherited  │ Inherited  │
│ Blending     │ Ensemble   │ Inherited  │ Inherited  │ Inherited  │
└──────────────┴────────────┴────────────┴────────────┴────────────┘
```

---

## Phase 3: Cross-Validation (COMPLETE)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         CROSS-VALIDATION PIPELINE                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                        PURGED K-FOLD CV                                  │    │
│  │                                                                          │    │
│  │  Standard K-Fold (WRONG):                                               │    │
│  │  |--Train--|--Test--|--Train--|  ← Future data leaks into training      │    │
│  │                                                                          │    │
│  │  Purged K-Fold (CORRECT):                                               │    │
│  │  |--Train--|PURGE|--Test--|EMBARGO|--Train--|                           │    │
│  │             60 bars       1440 bars                                      │    │
│  │                                                                          │    │
│  │  • Purge: Remove samples whose labels depend on test set                │    │
│  │  • Embargo: Break serial correlation after test set                     │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                      │                                           │
│                                      ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                    WALK-FORWARD FEATURE SELECTION                        │    │
│  │                                                                          │    │
│  │  For each fold:                                                         │    │
│  │    1. Compute importance on TRAINING data only (MDA/MDI)                │    │
│  │    2. Select top 50 features                                            │    │
│  │    3. Track feature frequency across folds                              │    │
│  │                                                                          │    │
│  │  Stable Features = Selected in >= 60% of folds                          │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                      │                                           │
│                                      ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                    OPTUNA HYPERPARAMETER TUNING                          │    │
│  │                                                                          │    │
│  │  For each trial (50 trials):                                            │    │
│  │    1. Sample hyperparameters from search space                          │    │
│  │    2. Run CV with sampled params                                        │    │
│  │    3. Compute mean F1 across folds                                      │    │
│  │    4. Apply variance penalty: score = mean_F1 - 0.1 * std_F1            │    │
│  │                                                                          │    │
│  │  Output: Best hyperparameters per (model, horizon)                      │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                      │                                           │
│                                      ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                    OUT-OF-FOLD (OOF) PREDICTIONS                         │    │
│  │                                                                          │    │
│  │  For each fold:                                                         │    │
│  │    1. Train model on fold's training data                               │    │
│  │    2. Predict on fold's validation data                                 │    │
│  │    3. Store predictions (never seen during training)                    │    │
│  │                                                                          │    │
│  │  Result: Every sample has a truly out-of-sample prediction              │    │
│  │          → This becomes Phase 4 meta-learner training data              │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                      │                                           │
│                                      ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                       STACKING DATASET                                   │    │
│  │                                                                          │    │
│  │  For each sample:                                                       │    │
│  │    • xgboost_prob_short, xgboost_prob_neutral, xgboost_prob_long        │    │
│  │    • lightgbm_prob_short, lightgbm_prob_neutral, lightgbm_prob_long     │    │
│  │    • lstm_prob_short, lstm_prob_neutral, lstm_prob_long                 │    │
│  │    • models_agree (boolean)                                             │    │
│  │    • avg_confidence                                                     │    │
│  │    • avg_entropy                                                        │    │
│  │    • y_true (actual label)                                              │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘

CV CONFIGURATION BY MODEL FAMILY:
┌──────────────┬──────────┬────────────┬─────────────────┐
│    Family    │  Folds   │  Trials    │  Training Cost  │
├──────────────┼──────────┼────────────┼─────────────────┤
│ Boosting     │ 5        │ 100        │ Low             │
│ Neural       │ 3        │ 50         │ Moderate        │
│ Transformer  │ 3        │ 30         │ High            │
└──────────────┴──────────┴────────────┴─────────────────┘
```

---

## CLI & Scripts

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              COMMAND INTERFACE                                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  PHASE 1 - Data Pipeline:                                                       │
│  ┌────────────────────────────────────────────────────────────────────────┐     │
│  │  ./pipeline run --symbols MES --horizons 5,10,15,20                    │     │
│  │                                                                         │     │
│  │  Options:                                                               │     │
│  │    --timeframe 5min          Target resampling timeframe               │     │
│  │    --feature-set core_full   Feature set selection                     │     │
│  │    --enable-wavelets         Enable wavelet decomposition              │     │
│  │    --mtf-mode both           Multi-timeframe mode                      │     │
│  │    --scaler-type robust      Scaling method                            │     │
│  │    --purge-bars 60           Purge window                              │     │
│  │    --embargo-bars 1440       Embargo window                            │     │
│  └────────────────────────────────────────────────────────────────────────┘     │
│                                                                                  │
│  PHASE 2 - Model Training:                                                      │
│  ┌────────────────────────────────────────────────────────────────────────┐     │
│  │  python scripts/train_model.py --model lstm --horizon 20               │     │
│  │                                                                         │     │
│  │  Options:                                                               │     │
│  │    --model <name>            Model: xgboost, lightgbm, catboost,       │     │
│  │                              lstm, gru, tcn, random_forest, logistic,  │     │
│  │                              svm, voting, stacking, blending (12 total)│     │
│  │    --horizon 5|10|15|20      Label horizon                             │     │
│  │    --batch-size 256          Training batch size                       │     │
│  │    --max-epochs 100          Maximum epochs                            │     │
│  │    --seq-len 60              Sequence length (neural)                  │     │
│  │    --hidden-size 128         Hidden units (neural)                     │     │
│  │    --device cuda|cpu         Training device                           │     │
│  │    --list-models             Show available models (should show 12)    │     │
│  │    --base-models x,y,z       Base models for ensembles                 │     │
│  └────────────────────────────────────────────────────────────────────────┘     │
│                                                                                  │
│  PHASE 3 - Cross-Validation:                                                    │
│  ┌────────────────────────────────────────────────────────────────────────┐     │
│  │  python scripts/run_cv.py --models xgboost,lstm --horizons 5,10,20    │     │
│  │                                                                         │     │
│  │  Options:                                                               │     │
│  │    --models all              Run all registered models                 │     │
│  │    --horizons all            Run all horizons                          │     │
│  │    --n-splits 5              Number of CV folds                        │     │
│  │    --tune                    Enable Optuna tuning                      │     │
│  │    --n-trials 50             Optuna trials per model                   │     │
│  │    --no-feature-selection    Disable walk-forward selection            │     │
│  │    --n-features 50           Features to select                        │     │
│  └────────────────────────────────────────────────────────────────────────┘     │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Directory Structure

```
/home/jake/Desktop/Research/
│
├── src/
│   ├── phase1/                      # PHASE 1: Data Pipeline
│   │   ├── stages/
│   │   │   ├── ingest/              # Data ingestion & validation
│   │   │   ├── clean/               # Resampling & gap handling
│   │   │   ├── features/            # 150+ technical indicators
│   │   │   ├── mtf/                 # Multi-timeframe features
│   │   │   ├── labeling/            # Triple-barrier labeling
│   │   │   ├── ga_optimize/         # Optuna barrier optimization
│   │   │   ├── final_labels/        # Quality scoring
│   │   │   ├── splits/              # Train/val/test splits
│   │   │   ├── scaling/             # Feature scaling
│   │   │   ├── datasets/            # TimeSeriesDataContainer
│   │   │   ├── validation/          # Quality checks
│   │   │   └── reporting/           # Summary reports
│   │   ├── config/                  # Feature configs
│   │   └── pipeline_config.py       # PipelineConfig dataclass
│   │
│   ├── models/                      # PHASE 2: Model Factory (12 models)
│   │   ├── base.py                  # BaseModel ABC
│   │   ├── registry.py              # ModelRegistry plugin system
│   │   ├── config.py                # TrainerConfig
│   │   ├── trainer.py               # Training orchestration
│   │   ├── device.py                # GPU utilities
│   │   ├── boosting/                # XGBoost, LightGBM, CatBoost
│   │   │   ├── xgboost_model.py
│   │   │   ├── lightgbm_model.py
│   │   │   └── catboost_model.py
│   │   ├── neural/                  # LSTM, GRU, TCN
│   │   │   ├── base_rnn.py          # Shared RNN training loop
│   │   │   ├── lstm_model.py
│   │   │   ├── gru_model.py
│   │   │   └── tcn_model.py
│   │   ├── classical/               # Random Forest, Logistic, SVM
│   │   │   ├── random_forest_model.py
│   │   │   ├── logistic_model.py
│   │   │   └── svm_model.py
│   │   └── ensemble/                # Voting, Stacking, Blending
│   │       ├── voting_model.py
│   │       ├── stacking_model.py
│   │       └── blending_model.py
│   │
│   ├── cross_validation/            # PHASE 3: CV System
│   │   ├── purged_kfold.py          # PurgedKFold, ModelAwareCV
│   │   ├── feature_selector.py      # Walk-forward MDA/MDI
│   │   ├── oof_generator.py         # OOF predictions, stacking
│   │   ├── cv_runner.py             # CrossValidationRunner
│   │   └── param_spaces.py          # Hyperparameter search spaces
│   │
│   └── cli/                         # Command-line interface
│       ├── run_commands.py          # run, rerun commands
│       └── status_commands.py       # status, validate commands
│
├── scripts/
│   ├── train_model.py               # Phase 2 training script
│   └── run_cv.py                    # Phase 3 CV script
│
├── config/
│   └── models/                      # YAML model configs (12 total)
│       ├── xgboost.yaml
│       ├── lightgbm.yaml
│       ├── lstm.yaml
│       ├── random_forest.yaml
│       ├── voting.yaml
│       └── ... (7 more configs)
│
├── notebooks/                       # Jupyter/Colab notebooks
│   ├── 01_quickstart.ipynb          # Complete pipeline walkthrough
│   ├── 02_train_all_models.ipynb    # Train all 12 models
│   ├── 03_cross_validation.ipynb    # CV and hyperparameter tuning
│   └── Phase1_Pipeline_Colab.ipynb  # Phase 1 on Colab with GPU
│
├── data/
│   ├── raw/                         # Input: {SYMBOL}_1m.parquet
│   ├── clean/                       # Resampled data
│   ├── features/                    # Feature-engineered data
│   ├── final/                       # Labeled data
│   ├── splits/
│   │   ├── scaled/                  # Scaled train/val/test
│   │   └── datasets/                # TimeSeriesDataContainer
│   └── stacking/                    # Phase 3 OOF predictions
│
├── experiments/
│   └── runs/                        # Phase 2 trained models
│       └── {run_id}/
│           ├── checkpoints/
│           ├── metrics/
│           └── config/
│
├── tests/                           # 1592 passing, 13 skipped
│   ├── models/                      # Phase 2 model tests
│   │   ├── test_boosting_models.py  # XGBoost, LightGBM, CatBoost
│   │   ├── test_neural_models.py    # LSTM, GRU, TCN
│   │   ├── test_classical_models.py # Random Forest, Logistic, SVM
│   │   ├── test_ensemble_models.py  # Voting, Stacking, Blending
│   │   └── test_trainer.py          # Unified trainer
│   ├── cross_validation/            # Phase 3 CV tests
│   │   ├── test_purged_kfold.py     # PurgedKFold, ModelAwareCV
│   │   ├── test_feature_selector.py # MDA/MDI selection
│   │   ├── test_oof_generator.py    # OOF predictions
│   │   └── test_cv_runner.py        # CV orchestration
│   └── utils/                       # Utility tests
│       └── test_notebook.py         # Notebook utility tests
│
├── CLAUDE.md                        # Project instructions
└── ARCHITECTURE_MAP.md              # This file
```

---

## Test Coverage

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              TEST SUMMARY                                        │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  Phase 2 Tests (tests/models/):                                                 │
│  ├── test_registry.py          32 tests   Model registration                   │
│  ├── test_boosting_models.py   58 tests   XGBoost, LightGBM, CatBoost          │
│  ├── test_neural_models.py     60 tests   LSTM, GRU, TCN                       │
│  ├── test_classical_models.py  47 tests   Random Forest, Logistic, SVM         │
│  ├── test_ensemble_models.py   52 tests   Voting, Stacking, Blending           │
│  ├── test_trainer.py           39 tests   Trainer orchestration                │
│  └── conftest.py               Fixtures   Synthetic data, mocks                │
│                                                                                  │
│  Phase 3 Tests (tests/cross_validation/):                                       │
│  ├── test_purged_kfold.py      36 tests   CV fold generation                   │
│  ├── test_feature_selector.py  29 tests   MDA/MDI selection                    │
│  ├── test_oof_generator.py     31 tests   OOF predictions                      │
│  ├── test_cv_runner.py         29 tests   CV orchestration                     │
│  └── conftest.py               Fixtures   Time series data, mocks              │
│                                                                                  │
│  Utilities (tests/utils/):                                                      │
│  ├── test_notebook.py          48 tests   Notebook plotting/utilities          │
│  └── conftest.py               Fixtures   Notebook mocks                       │
│                                                                                  │
│  Total: 1592 tests passing, 13 skipped (CatBoost env issues)                   │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘

Run all tests:
  python -m pytest tests/ -v

Run specific model family:
  python -m pytest tests/models/test_classical_models.py -v
  python -m pytest tests/models/test_ensemble_models.py -v
```

---

## Quick Reference

```
# Phase 1: Run data pipeline
./pipeline run --symbols MES --horizons 5,10,15,20

# Phase 2: Train single model
python scripts/train_model.py --model xgboost --horizon 20
python scripts/train_model.py --model lstm --horizon 20 --seq-len 60

# Phase 3: Run cross-validation
python scripts/run_cv.py --models xgboost,lightgbm --horizons 20
python scripts/run_cv.py --models all --tune --n-trials 50

# List available models
python scripts/train_model.py --list-models

# GPU info
python -c "from src.models.device import get_gpu_info; print(get_gpu_info())"
```

---

## Key Design Principles

1. **Plugin Architecture** - New models via `@register` decorator
2. **Time-Series Aware** - Purge/embargo prevents data leakage
3. **Walk-Forward** - Feature selection uses only past data
4. **GPU-First** - Mixed precision (bfloat16) for RTX 4070 Ti
5. **Quality Weighting** - Sample weights (0.5x-1.5x) for training
6. **Modular** - Each phase has clear inputs/outputs
7. **Tested** - 1592 tests covering all 12 models and pipelines
8. **Notebook Support** - Jupyter/Colab notebooks for interactive training

---

*Generated: 2025-12-25*
