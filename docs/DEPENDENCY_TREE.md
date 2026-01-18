# ML FACTORY DEPENDENCY TREE

```
================================================================================
                    ML FACTORY DEPENDENCY TREE
                    ==========================

Architecture Version: 4.0 (16-Stage Pipeline with Optuna Optimization)
Total Optuna Trials: 2,550

================================================================================

                           TABLE OF CONTENTS

1. Pipeline Overview
2. Complete 16-Stage Dependency Tree
3. Data Flow Diagram
4. Module Dependency Graph
5. Configuration Mapping
6. Migration Map (Current → Proposed)
7. Cross-Stage Dependencies

================================================================================
                         1. PIPELINE OVERVIEW
================================================================================

MLFactory Entry Point
─────────────────────
Entry: src/pipeline/unified.py (PROPOSED)
       src/pipeline/ml_factory.py (CURRENT)

Pipeline Phases:
  Phase A: Data Preparation     (Stages 1-6)
  Phase B: Optuna Optimization  (Stages 7-9)
  Phase C: Preprocessing        (Stages 10-12)
  Phase D: Training             (Stages 13-15)
  Phase E: Deployment           (Stage 16)

Total Optuna Trials: 2,550
  - Stage 7:  100 trials (label optimization)
  - Stage 8:  100 trials (feature selection)
  - Stage 9:   50 trials (feature pruning)
  - Stage 13: 2,300 trials (100 per model × 23 models)

================================================================================
                  2. COMPLETE 16-STAGE DEPENDENCY TREE
================================================================================

MLFactory (Entry Point)
│
├── [PHASE A: DATA PREPARATION - Stages 1-6]
│   │
│   ├── Stage 1: INGESTION
│   │   ├── depends_on: [raw OHLCV data]
│   │   ├── inputs:
│   │   │   └── data/raw/{symbol}_1m.parquet
│   │   ├── outputs:
│   │   │   └── Raw OHLCV DataFrame (in memory)
│   │   ├── config: config/pipeline/training.yaml (data section)
│   │   ├── current_location: src/phase1/stages/ingest/
│   │   │   ├── __init__.py
│   │   │   ├── loaders.py
│   │   │   ├── run.py
│   │   │   └── transformers.py
│   │   ├── proposed_location: src/pipeline/phases/data.py::Stage1Ingestion
│   │   └── runtime: ~1 second
│   │
│   ├── Stage 2: CLEANING
│   │   ├── depends_on: [Stage 1 - Ingestion]
│   │   ├── inputs:
│   │   │   └── Raw OHLCV DataFrame from Stage 1
│   │   ├── outputs:
│   │   │   └── data/processed/{symbol}_1m_clean.parquet
│   │   ├── config: config/pipeline/training.yaml
│   │   ├── current_location: src/phase1/stages/clean/
│   │   │   ├── __init__.py
│   │   │   ├── cleaner.py
│   │   │   ├── gap_handler.py
│   │   │   ├── utils.py
│   │   │   └── bar_builders/
│   │   │       ├── base.py
│   │   │       ├── dollar_bars.py
│   │   │       ├── time_bars.py
│   │   │       ├── volume_bars.py
│   │   │       └── factory.py
│   │   ├── proposed_location: src/pipeline/phases/data.py::Stage2Cleaning
│   │   └── runtime: ~2 seconds
│   │
│   ├── Stage 3: SESSIONS
│   │   ├── depends_on: [Stage 2 - Cleaning]
│   │   ├── inputs:
│   │   │   └── data/processed/{symbol}_1m_clean.parquet
│   │   ├── outputs:
│   │   │   └── Session-filtered DataFrame (in memory)
│   │   ├── config: config/pipeline/training.yaml (sessions section)
│   │   ├── current_location: src/phase1/stages/sessions/
│   │   ├── proposed_location: src/pipeline/phases/data.py::Stage3Sessions
│   │   └── runtime: ~1 second
│   │
│   ├── Stage 4: MTF UPSCALING
│   │   ├── depends_on: [Stage 3 - Sessions]
│   │   ├── inputs:
│   │   │   └── Session-filtered 1-min OHLCV
│   │   ├── outputs:
│   │   │   └── data/processed/{symbol}_{tf}.parquet (9 files)
│   │   │       └── Timeframes: 1m, 5m, 10m, 15m, 20m, 25m, 30m, 45m, 1h
│   │   ├── config: config/features/mtf_strategies.yaml
│   │   ├── current_location: src/phase1/stages/mtf/
│   │   │   ├── __init__.py
│   │   │   ├── upscaler.py
│   │   │   ├── resampler.py
│   │   │   └── alignment.py
│   │   ├── proposed_location: src/pipeline/phases/data.py::Stage4MTFUpscaling
│   │   └── runtime: ~4 seconds
│   │
│   ├── Stage 5: FEATURES
│   │   ├── depends_on: [Stage 4 - MTF Upscaling]
│   │   ├── inputs:
│   │   │   └── MTF OHLCV DataFrames (9 timeframes)
│   │   ├── outputs:
│   │   │   └── data/features/{symbol}_features.parquet (~180 features)
│   │   ├── config: config/features/model_features.yaml
│   │   ├── feature_families (12 total, 162+ features):
│   │   │   ├── Momentum (23): RSI, MACD, Stochastic, Williams %R, ROC, CCI, MFI
│   │   │   ├── Moving Averages (16): SMA, EMA at multiple periods + crossovers
│   │   │   ├── Volatility (25): ATR, Bollinger, Keltner, HV, Parkinson, GARCH
│   │   │   ├── Volume (15): OBV, VWAP, TWAP, Dollar Volume, Volume Ratio
│   │   │   ├── Trend (6): ADX, +DI/-DI, Supertrend
│   │   │   ├── Price (12): Returns, ratios, autocorrelation, CLV
│   │   │   ├── Microstructure (15): Amihud, Roll, Kyle, spreads, imbalances
│   │   │   ├── Entropy (12): Shannon, Lempel-Ziv, ApEn, SampEn, Hurst
│   │   │   ├── Wavelets (15): DWT coefficients, energy, entropy
│   │   │   ├── Temporal (9): Hour/DOW sin/cos, session progress
│   │   │   ├── Regime (9): Volatility regime, trend regime, composite
│   │   │   └── MTF Indicators (30+): Cross-timeframe features
│   │   ├── current_location: src/phase1/stages/features/
│   │   │   ├── cli.py
│   │   │   ├── constants.py
│   │   │   ├── momentum.py
│   │   │   ├── moving_averages.py
│   │   │   ├── nan_handling.py
│   │   │   ├── numba_functions.py
│   │   │   ├── price_features.py
│   │   │   ├── regime.py
│   │   │   ├── scaling.py
│   │   │   ├── temporal.py
│   │   │   ├── trend.py
│   │   │   └── wavelets.py
│   │   ├── proposed_location: src/pipeline/phases/data.py::Stage5Features
│   │   └── runtime: ~16 seconds
│   │
│   └── Stage 6: REGIME DETECTION
│       ├── depends_on: [Stage 5 - Features]
│       ├── inputs:
│       │   └── data/features/{symbol}_features.parquet
│       ├── outputs:
│       │   └── Regime-labeled DataFrame (in memory)
│       │       └── 9 composite regimes (vol_regime × trend_regime)
│       ├── config: config/features/model_features.yaml (regime section)
│       ├── current_location: src/phase1/stages/regime/
│       ├── proposed_location: src/pipeline/phases/data.py::Stage6Regime
│       └── runtime: ~2 seconds
│
│   [CHECKPOINT: data/features/{symbol}_features.parquet]
│
├── [PHASE B: OPTUNA OPTIMIZATION - Stages 7-9]
│   │
│   ├── Stage 7: OPTUNA LABEL OPTIMIZATION (Triple Barrier)
│   │   ├── depends_on: [Stage 6 - Regime Detection]
│   │   ├── inputs:
│   │   │   └── data/features/{symbol}_features.parquet
│   │   ├── outputs:
│   │   │   ├── optimal_barrier_params.json
│   │   │   ├── label_optimization_history.csv
│   │   │   └── Labeled DataFrame with optimized triple-barrier labels
│   │   ├── optuna_trials: 100
│   │   ├── search_space:
│   │   │   ├── upper_mult: [1.0, 4.0] - Upper barrier ATR multiplier
│   │   │   ├── lower_mult: [1.0, 4.0] - Lower barrier ATR multiplier
│   │   │   ├── horizon: [5, 60] - Maximum holding bars
│   │   │   └── atr_period: [7, 28] - ATR calculation period
│   │   ├── objective: Maximize label quality via quick CV (LightGBM)
│   │   ├── config: config/optimization/label_optimization.yaml
│   │   ├── current_location: src/phase1/stages/labeling/
│   │   │   └── triple_barrier.py
│   │   ├── also_uses: src/labeling/
│   │   ├── proposed_location: src/pipeline/phases/optimization.py::Stage7LabelOptimization
│   │   └── runtime: ~10 minutes
│   │
│   ├── Stage 8: OPTUNA FEATURE SELECTION
│   │   ├── depends_on: [Stage 7 - Label Optimization]
│   │   ├── inputs:
│   │   │   └── Labeled DataFrame from Stage 7 (~180 features)
│   │   ├── outputs:
│   │   │   ├── optimal_feature_mask.json
│   │   │   ├── selected_features.txt (~60-100 features)
│   │   │   └── Feature-selected DataFrame
│   │   ├── optuna_trials: 100
│   │   ├── search_space:
│   │   │   └── Binary include/exclude for each of 162+ features
│   │   ├── objective: Maximize F1 with minimal feature set
│   │   ├── config: config/optimization/feature_selection.yaml
│   │   ├── current_location: src/features/optimization.py
│   │   │   └── (Also: src/feature_selection/)
│   │   ├── proposed_location: src/pipeline/phases/optimization.py::Stage8FeatureSelection
│   │   └── runtime: ~15 minutes
│   │
│   └── Stage 9: OPTUNA FEATURE PRUNING
│       ├── depends_on: [Stage 8 - Feature Selection]
│       ├── inputs:
│       │   └── Selected features from Stage 8 (~60-100 features)
│       ├── outputs:
│       │   ├── pruned_feature_mask.json
│       │   ├── feature_importance_ranking.csv
│       │   └── Pruned feature set (~30-60 features)
│       ├── optuna_trials: 50
│       ├── search_space:
│       │   ├── importance_threshold: [0.001, 0.1] (log scale)
│       │   ├── top_k_features: [20, 100]
│       │   └── importance_method: [gain, split, shap]
│       ├── objective: Remove low-importance features
│       ├── config: config/optimization/feature_pruning.yaml
│       ├── current_location: src/features/optimization.py
│       ├── proposed_location: src/pipeline/phases/optimization.py::Stage9FeaturePruning
│       └── runtime: ~8 minutes
│
│   [CHECKPOINT: data/optimized/{symbol}_optimized.parquet]
│
├── [PHASE C: PREPROCESSING - Stages 10-12]
│   │
│   ├── Stage 10: SPLITS
│   │   ├── depends_on: [Stage 9 - Feature Pruning]
│   │   ├── inputs:
│   │   │   └── data/optimized/{symbol}_optimized.parquet
│   │   ├── outputs:
│   │   │   └── Train/Val/Test DataFrames (in memory)
│   │   ├── split_ratios:
│   │   │   ├── train: 70%
│   │   │   ├── val: 15%
│   │   │   └── test: 15%
│   │   ├── leakage_prevention:
│   │   │   ├── purge_bars: 60 (removes label overlap)
│   │   │   └── embargo_bars: 1440 (~5 days, prevents serial correlation)
│   │   ├── config: config/pipeline/training.yaml (data section)
│   │   ├── current_location: src/phase1/stages/splits/
│   │   ├── proposed_location: src/pipeline/phases/training.py::Stage10Splits
│   │   └── runtime: ~1 second
│   │
│   ├── Stage 11: SCALING
│   │   ├── depends_on: [Stage 10 - Splits]
│   │   ├── inputs:
│   │   │   └── Train/Val/Test DataFrames
│   │   ├── outputs:
│   │   │   ├── data/splits/scaled/{symbol}_train.parquet
│   │   │   ├── data/splits/scaled/{symbol}_val.parquet
│   │   │   ├── data/splits/scaled/{symbol}_test.parquet
│   │   │   └── scaler.pkl (RobustScaler, fit on train only)
│   │   ├── methodology: Train-only fitting, transform all splits
│   │   ├── config: config/pipeline/training.yaml
│   │   ├── current_location: src/phase1/stages/scaling/
│   │   ├── proposed_location: src/pipeline/phases/training.py::Stage11Scaling
│   │   └── runtime: ~1 second
│   │
│   └── Stage 12: ADAPTATION (Model-Family Adapters)
│       ├── depends_on: [Stage 11 - Scaling]
│       ├── inputs:
│       │   └── data/splits/scaled/{symbol}_{split}.parquet
│       ├── outputs:
│       │   └── Model-specific tensors (in memory)
│       │       ├── Tabular: 2D arrays (N, ~60)
│       │       ├── Sequence: 3D windows (N, T, ~60)
│       │       └── Multi-Resolution: 4D tensors (N, 9, T, 4)
│       ├── adapters:
│       │   ├── TabularAdapter
│       │   │   ├── output_shape: (N, F)
│       │   │   └── models: XGBoost, LightGBM, CatBoost, RF, Logistic, SVM
│       │   ├── SequenceAdapter
│       │   │   ├── output_shape: (N, T, F)
│       │   │   └── models: LSTM, GRU, TCN, Transformer
│       │   └── MultiResolutionAdapter
│       │       ├── output_shape: (N, TF, T, 4)
│       │       └── models: PatchTST, iTransformer, TFT, N-BEATS, InceptionTime, ResNet1D
│       ├── config: config/models.yaml (adapters section)
│       ├── current_location: src/phase1/stages/datasets/adapters/
│       │   ├── __init__.py
│       │   ├── multi_resolution.py (619 lines - COMPLETE)
│       │   └── utils.py
│       ├── also_in: src/adapters/
│       ├── proposed_location: src/pipeline/phases/training.py::Stage12Adaptation
│       └── runtime: ~2 seconds
│
│   [CHECKPOINT: data/splits/scaled/{symbol}_{split}.parquet]
│
├── [PHASE D: TRAINING - Stages 13-15]
│   │
│   ├── Stage 13: OPTUNA HYPERPARAMETER OPTIMIZATION
│   │   ├── depends_on: [Stage 12 - Adaptation]
│   │   ├── inputs:
│   │   │   └── Adapted tensors from Stage 12
│   │   ├── outputs:
│   │   │   ├── {model}_best_params.json (per model)
│   │   │   ├── {model}_optimization_history.csv (per model)
│   │   │   └── Optimal hyperparameters for all 23 models
│   │   ├── optuna_trials: 100 per model (2,300 total)
│   │   ├── models_optimized (23 total):
│   │   │   ├── Boosting (3): XGBoost, LightGBM, CatBoost
│   │   │   ├── Classical (3): Random Forest, Logistic, SVM
│   │   │   ├── Neural Basic (4): LSTM, GRU, TCN, Transformer
│   │   │   ├── Neural Advanced (6): PatchTST, iTransformer, TFT, N-BEATS, InceptionTime, ResNet1D
│   │   │   ├── Ensemble (3): Voting, Stacking, Blending
│   │   │   └── Meta-Learners (4): Ridge Meta, MLP Meta, Calibrated Meta, XGBoost Meta
│   │   ├── search_spaces_location: src/optimization/search_spaces.py
│   │   ├── config: config/optimization/hyperparameter.yaml
│   │   ├── current_location: src/optimization/
│   │   ├── proposed_location: src/pipeline/phases/training.py::Stage13HyperparameterOptimization
│   │   └── runtime: ~20-60 minutes per model
│   │
│   ├── Stage 14: TRAINING (Base Models)
│   │   ├── depends_on: [Stage 13 - Hyperparameter Optimization]
│   │   ├── inputs:
│   │   │   ├── Adapted tensors from Stage 12
│   │   │   └── Optimal hyperparameters from Stage 13
│   │   ├── outputs:
│   │   │   ├── experiments/runs/{run_id}/models/{model}_{symbol}_h{horizon}.pkl
│   │   │   ├── experiments/runs/{run_id}/models/{model}_{symbol}_h{horizon}.pt (neural)
│   │   │   ├── experiments/runs/{run_id}/reports/{model}_report.json
│   │   │   └── OOF predictions for each model (for Stage 15)
│   │   ├── training_method: 5-fold PurgedKFold CV with OOF generation
│   │   ├── model_families:
│   │   │   ├── Boosting (3 models):
│   │   │   │   ├── src/models/boosting/xgboost_model.py
│   │   │   │   ├── src/models/boosting/lightgbm_model.py
│   │   │   │   └── src/models/boosting/catboost_model.py
│   │   │   ├── Classical (3 models):
│   │   │   │   ├── src/models/classical/random_forest.py
│   │   │   │   ├── src/models/classical/logistic.py
│   │   │   │   └── src/models/classical/svm.py
│   │   │   └── Neural (10 models):
│   │   │       ├── src/models/neural/lstm_model.py
│   │   │       ├── src/models/neural/gru_model.py
│   │   │       ├── src/models/neural/tcn_model.py
│   │   │       ├── src/models/neural/transformer.py (basic)
│   │   │       ├── src/models/neural/patchtst_model.py
│   │   │       ├── src/models/neural/itransformer_model.py
│   │   │       ├── src/models/neural/tft_model.py
│   │   │       └── (N-BEATS, InceptionTime, ResNet1D)
│   │   ├── config: config/models/{model_name}.yaml (23 files)
│   │   ├── current_location: src/models/
│   │   │   ├── base.py (BaseModel interface)
│   │   │   ├── registry.py (Model registry)
│   │   │   ├── trainer.py (Unified trainer)
│   │   │   └── training/ (Training utilities)
│   │   ├── proposed_location: src/pipeline/phases/training.py::Stage14Training
│   │   └── runtime: 15s-5min per model
│   │
│   └── Stage 15: STACKING (Meta-Learner)
│       ├── depends_on: [Stage 14 - Training]
│       ├── inputs:
│       │   ├── OOF predictions from Stage 14 base models
│       │   └── Trained base models from Stage 14
│       ├── outputs:
│       │   ├── experiments/runs/{run_id}/models/meta_learners/
│       │   │   ├── ridge_meta_{symbol}_h{horizon}.pkl
│       │   │   ├── mlp_meta_{symbol}_h{horizon}.pt
│       │   │   ├── calibrated_meta_{symbol}_h{horizon}.pkl
│       │   │   └── xgboost_meta_{symbol}_h{horizon}.pkl
│       │   └── Stacking ensemble performance report
│       ├── meta_learners (4):
│       │   ├── ridge_meta: L2-regularized linear stacking
│       │   ├── mlp_meta: Small neural network meta-learner
│       │   ├── calibrated_meta: Isotonic/Platt scaling
│       │   └── xgboost_meta: Gradient boosting meta-learner
│       ├── base_model_selection: 3-4 heterogeneous (1 per family)
│       │   ├── Tabular: CatBoost OR LightGBM
│       │   ├── CNN: TCN
│       │   ├── Transformer: PatchTST OR TFT
│       │   └── Optional 4th: N-BEATS OR Ridge
│       ├── optuna_trials: 50 (for meta-learner hyperparameters)
│       ├── config: config/meta_learner.yaml
│       ├── current_location:
│       │   ├── src/cross_validation/oof_stacking.py
│       │   └── src/models/ensemble/meta_learners/
│       ├── proposed_location: src/pipeline/phases/training.py::Stage15Stacking
│       └── runtime: ~5 minutes
│
│   [CHECKPOINT: experiments/runs/{run_id}/models/]
│
└── [PHASE E: DEPLOYMENT - Stage 16]
    │
    └── Stage 16: BUNDLING
        ├── depends_on: [Stage 15 - Stacking]
        ├── inputs:
        │   ├── Trained models from Stage 14-15
        │   ├── Fitted scaler from Stage 11
        │   ├── Feature mask from Stages 8-9
        │   └── Barrier params from Stage 7
        ├── outputs:
        │   └── experiments/runs/{run_id}/bundles/{model}_bundle.pkl
        │       └── ModelBundle V1.1.0 contents:
        │           ├── Trained model weights (.pkl/.pt)
        │           ├── Fitted RobustScaler
        │           ├── Feature mask (optimized)
        │           ├── Barrier parameters (optimized)
        │           ├── Optimal hyperparameters
        │           ├── PreprocessingGraph (feature lineage)
        │           └── BundleMetadata
        ├── config: config/pipeline/training.yaml (experiment section)
        ├── current_location: src/models/training/ (partial)
        ├── proposed_location: src/pipeline/phases/deployment.py::Stage16Bundling
        └── runtime: ~5 seconds per model


================================================================================
                        3. DATA FLOW DIAGRAM
================================================================================

                        DATA FLOW ACROSS STAGES

Raw Data                                         Trained Models
   │                                                   ▲
   ▼                                                   │
┌──────────────────────────────────────────────────────┴───────────────────────┐
│                                                                               │
│  STAGE 1         STAGE 2         STAGE 3         STAGE 4         STAGE 5     │
│  Ingest   ───►   Clean    ───►   Sessions ───►   MTF      ───►   Features    │
│  1s              2s              1s              4s              16s          │
│                                                                               │
│  Raw OHLCV       Clean           Session        9 TF            162          │
│  parquet         parquet         filtered       parquets        features     │
│                                                                               │
└──────────────────────────────────┬───────────────────────────────────────────┘
                                   ▼
                            [CHECKPOINT 1]
                  data/features/{symbol}_features.parquet
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                          OPTUNA OPTIMIZATION                                  │
│                                                                               │
│  STAGE 6         STAGE 7         STAGE 8         STAGE 9                     │
│  Regime   ───►   Label    ───►   Feature  ───►   Feature                     │
│  2s              Opt             Selection       Pruning                      │
│                  10min           15min           8min                         │
│                  100 trials      100 trials      50 trials                    │
│                                                                               │
│  9 regimes       Optimal         ~60-100         ~30-60                       │
│                  barriers        features        features                     │
│                                                                               │
└──────────────────────────────────┬───────────────────────────────────────────┘
                                   ▼
                            [CHECKPOINT 2]
                 data/optimized/{symbol}_optimized.parquet
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                           PREPROCESSING                                       │
│                                                                               │
│  STAGE 10        STAGE 11        STAGE 12                                    │
│  Splits   ───►   Scaling  ───►   Adaptation                                  │
│  1s              1s              2s                                          │
│                                                                               │
│  70/15/15        Train-only      ┌─────────────────────────┐                 │
│  + purge         RobustScaler    │  Tabular: 2D (N, F)     │                 │
│  + embargo                       │  Sequence: 3D (N, T, F) │                 │
│                                  │  MultiRes: 4D (N,TF,T,4)│                 │
│                                  └─────────────────────────┘                 │
└──────────────────────────────────┬───────────────────────────────────────────┘
                                   ▼
                            [CHECKPOINT 3]
                data/splits/scaled/{symbol}_{split}.parquet
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                              TRAINING                                         │
│                                                                               │
│  STAGE 13                STAGE 14              STAGE 15                      │
│  Hyperparameter   ───►   Training      ───►   Stacking                       │
│  Optimization            + OOF                Meta-Learner                   │
│  100 trials/model        5-fold CV            50 trials                      │
│  (~20-60min/model)       (15s-5min/model)     5min                           │
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────┐         │
│  │                    23 MODELS TRAINED                             │         │
│  │  ┌───────────┬───────────┬───────────┬───────────┬───────────┐ │         │
│  │  │ Boosting  │ Classical │ Neural    │ Advanced  │ Meta-     │ │         │
│  │  │ (3)       │ (3)       │ Basic (4) │ (6)       │ Learners  │ │         │
│  │  ├───────────┼───────────┼───────────┼───────────┼───────────┤ │         │
│  │  │ XGBoost   │ RF        │ LSTM      │ PatchTST  │ Ridge     │ │         │
│  │  │ LightGBM  │ Logistic  │ GRU       │ iTrans    │ MLP       │ │         │
│  │  │ CatBoost  │ SVM       │ TCN       │ TFT       │ Calibr    │ │         │
│  │  │           │           │ Transf    │ N-BEATS   │ XGB Meta  │ │         │
│  │  │           │           │           │ IncepTime │           │ │         │
│  │  │           │           │           │ ResNet1D  │           │ │         │
│  │  └───────────┴───────────┴───────────┴───────────┴───────────┘ │         │
│  └─────────────────────────────────────────────────────────────────┘         │
│                                                                               │
└──────────────────────────────────┬───────────────────────────────────────────┘
                                   ▼
                            [CHECKPOINT 4]
                experiments/runs/{run_id}/models/
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                            DEPLOYMENT                                         │
│                                                                               │
│  STAGE 16                                                                     │
│  Bundling                                                                     │
│  5s/model                                                                     │
│                                                                               │
│  ModelBundle V1.1.0:                                                          │
│  ├── trained_model.pkl/.pt                                                    │
│  ├── scaler.pkl                                                               │
│  ├── feature_mask.json                                                        │
│  ├── barrier_params.json                                                      │
│  ├── hyperparameters.json                                                     │
│  ├── preprocessing_graph.pkl                                                  │
│  └── metadata.json                                                            │
│                                                                               │
└──────────────────────────────────┬───────────────────────────────────────────┘
                                   ▼
                            [FINAL OUTPUT]
              experiments/runs/{run_id}/bundles/{model}_bundle.pkl


================================================================================
                    4. MODULE DEPENDENCY GRAPH
================================================================================

                          INTERNAL MODULE DEPENDENCIES

src/pipeline/ (PROPOSED - Unified Entry Point)
├── depends_on:
│   ├── src/phase1/              # Data pipeline stages 1-6
│   ├── src/optimization/        # Optuna optimization stages 7-9, 13
│   ├── src/models/              # Training stages 14-15
│   ├── src/cross_validation/    # OOF generation, CV
│   ├── src/features/            # Feature optimization
│   └── src/config/              # Configuration loading

src/phase1/ (Data Pipeline)
├── stages/
│   ├── ingest/       → depends_on: [pandas, pyarrow]
│   ├── clean/        → depends_on: [src/phase1/stages/ingest]
│   ├── sessions/     → depends_on: [src/phase1/stages/clean]
│   ├── mtf/          → depends_on: [src/phase1/stages/sessions]
│   ├── features/     → depends_on: [src/phase1/stages/mtf, numpy, numba]
│   ├── regime/       → depends_on: [src/phase1/stages/features]
│   ├── labeling/     → depends_on: [src/phase1/stages/regime]
│   ├── splits/       → depends_on: [src/phase1/stages/labeling]
│   ├── scaling/      → depends_on: [src/phase1/stages/splits, sklearn]
│   └── datasets/
│       └── adapters/ → depends_on: [src/phase1/stages/scaling]

src/optimization/ (Optuna Optimization)
├── depends_on:
│   ├── optuna
│   ├── src/models/registry.py    # Model instantiation
│   ├── src/cross_validation/     # CV for evaluation
│   └── src/features/optimization.py

src/models/ (Model Training)
├── base.py           → depends_on: [abc, dataclasses]
├── registry.py       → depends_on: [src/models/base]
├── trainer.py        → depends_on: [src/models/registry, src/phase1/stages/datasets]
├── boosting/         → depends_on: [xgboost, lightgbm, catboost]
├── classical/        → depends_on: [sklearn]
├── neural/           → depends_on: [torch, pytorch-lightning]
└── ensemble/
    └── meta_learners/ → depends_on: [src/cross_validation/oof_stacking]

src/cross_validation/ (CV Infrastructure)
├── cpcv.py           → depends_on: [numpy, sklearn]
├── fold_scaling.py   → depends_on: [sklearn.preprocessing]
├── oof_io.py         → depends_on: [pandas, numpy]
├── oof_validation.py → depends_on: [src/cross_validation/oof_io]
├── param_spaces.py   → depends_on: [optuna]
├── pbo.py            → depends_on: [numpy, scipy]
├── sequence_cv.py    → depends_on: [sklearn.model_selection]
└── walk_forward.py   → depends_on: [src/cross_validation/sequence_cv]

src/features/ (Feature Engineering)
├── depends_on:
│   ├── src/phase1/stages/features/
│   └── optuna (for optimization)
└── optimization.py   → Stage 8-9 implementation


================================================================================
                      5. CONFIGURATION MAPPING
================================================================================

                    CONFIG FILE → STAGE MAPPING

┌─────────────────────────────────────────────────────────────────────────────┐
│                         GLOBAL CONFIGURATIONS                                │
├─────────────────────────────────────────────────────────────────────────────┤
│  config/global.yaml                                                          │
│  └── controls: All stages (paths, symbols, environment)                      │
│                                                                              │
│  config/labeling.yaml                                                        │
│  └── controls: Stage 7 (triple barrier defaults)                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                         PIPELINE CONFIGURATIONS                              │
├─────────────────────────────────────────────────────────────────────────────┤
│  config/pipeline/training.yaml                                               │
│  └── controls:                                                               │
│      ├── Stage 1-3: data paths, symbols                                      │
│      ├── Stage 10: split ratios, purge, embargo                              │
│      ├── Stage 11: scaling method                                            │
│      └── Stage 14: batch size, epochs, device                                │
│                                                                              │
│  config/pipeline/cv.yaml                                                     │
│  └── controls:                                                               │
│      ├── Stage 14: PurgedKFold settings                                      │
│      ├── Stage 15: OOF generation settings                                   │
│      └── Walk-forward validation                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                        FEATURE CONFIGURATIONS                                │
├─────────────────────────────────────────────────────────────────────────────┤
│  config/features/model_features.yaml                                         │
│  └── controls:                                                               │
│      ├── Stage 5: Feature family selection                                   │
│      └── Stage 12: Per-model feature strategies                              │
│                                                                              │
│  config/features/mtf_strategies.yaml                                         │
│  └── controls:                                                               │
│      ├── Stage 4: MTF timeframe ladder                                       │
│      └── Stage 12: MTF adapter configuration                                 │
│                                                                              │
│  config/features/selection_methods.yaml                                      │
│  └── controls: Stage 8-9 (feature selection methods)                         │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                       OPTIMIZATION CONFIGURATIONS                            │
├─────────────────────────────────────────────────────────────────────────────┤
│  config/optimization/label_optimization.yaml                                 │
│  └── controls: Stage 7                                                       │
│      ├── trials: 100                                                         │
│      └── search_space: upper_mult, lower_mult, horizon, atr_period          │
│                                                                              │
│  config/optimization/feature_selection.yaml                                  │
│  └── controls: Stage 8                                                       │
│      ├── trials: 100                                                         │
│      └── search_space: binary include/exclude                                │
│                                                                              │
│  config/optimization/feature_pruning.yaml                                    │
│  └── controls: Stage 9                                                       │
│      ├── trials: 50                                                          │
│      └── search_space: importance_threshold, top_k, method                   │
│                                                                              │
│  config/optimization/hyperparameter.yaml                                     │
│  └── controls: Stage 13                                                      │
│      ├── trials: 100 per model                                               │
│      └── search_space: model-specific hyperparameters                        │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                         MODEL CONFIGURATIONS                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│  config/models/{model_name}.yaml (23 files)                                  │
│  └── controls: Stage 13-14                                                   │
│      ├── model.name, model.family                                            │
│      ├── defaults (hyperparameters)                                          │
│      ├── training settings                                                   │
│      └── device settings                                                     │
│                                                                              │
│  Files:                                                                      │
│  ├── xgboost.yaml, lightgbm.yaml, catboost.yaml      (Boosting)              │
│  ├── random_forest.yaml, logistic.yaml, svm.yaml     (Classical)             │
│  ├── lstm.yaml, gru.yaml, tcn.yaml, transformer.yaml (Neural Basic)          │
│  ├── patchtst.yaml, itransformer.yaml, tft.yaml      (Neural Advanced)       │
│  ├── nbeats.yaml, inceptiontime.yaml, resnet1d.yaml  (Neural Advanced)       │
│  ├── voting.yaml, stacking.yaml, blending.yaml       (Ensemble)              │
│  └── ridge_meta.yaml, mlp_meta.yaml, etc.            (Meta-Learners)         │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                        ENSEMBLE CONFIGURATIONS                               │
├─────────────────────────────────────────────────────────────────────────────┤
│  config/ensembles/boosting_trio.yaml                                         │
│  └── controls: Stage 15 (homogeneous ensemble)                               │
│                                                                              │
│  config/ensembles/temporal_stack.yaml                                        │
│  └── controls: Stage 15 (sequence stacking)                                  │
│                                                                              │
│  config/meta_learner.yaml                                                    │
│  └── controls: Stage 15 (heterogeneous stacking)                             │
│      ├── base_models: {tabular, cnn, transformer, optional_4th}              │
│      ├── meta_learner: ridge_meta | mlp_meta | xgboost_meta | calibrated     │
│      ├── oof settings                                                        │
│      └── optuna settings (50 trials)                                         │
└─────────────────────────────────────────────────────────────────────────────┘


================================================================================
                      6. MIGRATION MAP
================================================================================

                    CURRENT → PROPOSED LOCATION MAPPING

┌─────────────────────────────────────────────────────────────────────────────┐
│                         DATA PIPELINE COMPONENTS                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  CURRENT LOCATION                    →    PROPOSED LOCATION                  │
│  ───────────────────────────────────────────────────────────────────────    │
│                                                                              │
│  src/phase1/stages/ingest/           →    src/pipeline/phases/data.py       │
│  src/phase1/stages/clean/            →    (Stage1Ingestion class)           │
│  src/phase1/stages/sessions/         →    (Stage2Cleaning class)            │
│  src/phase1/stages/mtf/              →    (Stage3Sessions class)            │
│  src/phase1/stages/features/         →    (Stage4MTFUpscaling class)        │
│  src/phase1/stages/regime/           →    (Stage5Features class)            │
│                                      →    (Stage6Regime class)              │
│                                                                              │
│  STATUS: KEEP ORIGINAL + CREATE WRAPPER                                      │
│  (Original modules stay for backwards compatibility)                         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                      OPTIMIZATION COMPONENTS                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  CURRENT LOCATION                    →    PROPOSED LOCATION                  │
│  ───────────────────────────────────────────────────────────────────────    │
│                                                                              │
│  src/phase1/stages/labeling/         →    src/pipeline/phases/optimization.py│
│  + src/labeling/                     →    (Stage7LabelOptimization class)   │
│                                                                              │
│  src/features/optimization.py        →    src/pipeline/phases/optimization.py│
│  + src/feature_selection/            →    (Stage8FeatureSelection class)    │
│                                      →    (Stage9FeaturePruning class)      │
│                                                                              │
│  STATUS: CONSOLIDATE INTO SINGLE MODULE                                      │
│  (src/features/optimization.py + src/feature_selection/ merge)              │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                        TRAINING COMPONENTS                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  CURRENT LOCATION                    →    PROPOSED LOCATION                  │
│  ───────────────────────────────────────────────────────────────────────    │
│                                                                              │
│  src/phase1/stages/splits/           →    src/pipeline/phases/training.py   │
│  src/phase1/stages/scaling/          →    (Stage10Splits class)             │
│  src/phase1/stages/datasets/adapters/→    (Stage11Scaling class)            │
│                                      →    (Stage12Adaptation class)         │
│                                                                              │
│  src/optimization/                   →    src/pipeline/phases/training.py   │
│  (search_spaces.py)                  →    (Stage13HyperparameterOptimization)│
│                                                                              │
│  src/models/trainer.py               →    src/pipeline/phases/training.py   │
│  src/models/training/                →    (Stage14Training class)           │
│  src/training/                       →    (TrainingOrchestrator wrapper)    │
│                                                                              │
│  src/cross_validation/oof_stacking.py→    src/pipeline/phases/training.py   │
│  src/models/ensemble/meta_learners/  →    (Stage15Stacking class)           │
│                                                                              │
│  STATUS: WRAP EXISTING + NEW ORCHESTRATION                                   │
│  (Keep src/models/, add orchestration layer)                                 │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                       DEPLOYMENT COMPONENTS                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  CURRENT LOCATION                    →    PROPOSED LOCATION                  │
│  ───────────────────────────────────────────────────────────────────────    │
│                                                                              │
│  src/models/training/                →    src/pipeline/phases/deployment.py │
│  (partial bundling)                  →    (Stage16Bundling class)           │
│                                                                              │
│  STATUS: NEW MODULE                                                          │
│  (Create ModelBundle class with full artifact packaging)                     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                     CONSOLIDATION SUMMARY                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  WHAT STAYS SEPARATE (Keep Original):                                        │
│  ├── src/models/boosting/          # XGBoost, LightGBM, CatBoost            │
│  ├── src/models/classical/         # RF, Logistic, SVM                       │
│  ├── src/models/neural/            # LSTM, GRU, TCN, Transformer, etc.      │
│  ├── src/models/ensemble/          # Voting, Stacking, Blending             │
│  ├── src/models/base.py            # BaseModel interface                     │
│  ├── src/models/registry.py        # Model registry                          │
│  ├── src/cross_validation/         # CV infrastructure                       │
│  └── src/phase1/stages/features/   # Feature computation functions          │
│                                                                              │
│  WHAT GETS CONSOLIDATED:                                                     │
│  ├── src/features/optimization.py + src/feature_selection/                   │
│  │   └── → src/pipeline/phases/optimization.py                               │
│  ├── src/training/ + src/models/trainer.py + src/models/training/            │
│  │   └── → src/pipeline/phases/training.py                                   │
│  └── Multiple labeling modules                                               │
│      └── → src/pipeline/phases/optimization.py::Stage7                       │
│                                                                              │
│  WHAT GETS CREATED NEW:                                                      │
│  ├── src/pipeline/unified.py       # MLPipeline master orchestrator          │
│  ├── src/pipeline/config.py        # MLConfig unified configuration          │
│  ├── src/pipeline/state.py         # PipelineState management                │
│  ├── src/pipeline/phases/data.py   # Stages 1-6 wrappers                     │
│  ├── src/pipeline/phases/optimization.py  # Stages 7-9                       │
│  ├── src/pipeline/phases/training.py      # Stages 10-15                     │
│  ├── src/pipeline/phases/deployment.py    # Stage 16                         │
│  └── src/cli/unified_cli.py        # Single 'ml' CLI                         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘


================================================================================
                    7. CROSS-STAGE DEPENDENCIES
================================================================================

                    ARTIFACT FLOW BETWEEN STAGES

Stage 1 (Ingestion)
└── produces: Raw DataFrame
    └── consumed_by: Stage 2

Stage 2 (Cleaning)
└── produces: data/processed/{symbol}_1m_clean.parquet
    └── consumed_by: Stage 3

Stage 3 (Sessions)
└── produces: Session-filtered DataFrame
    └── consumed_by: Stage 4

Stage 4 (MTF Upscaling)
└── produces: data/processed/{symbol}_{tf}.parquet (×9)
    └── consumed_by: Stage 5

Stage 5 (Features)
└── produces: data/features/{symbol}_features.parquet (~180 features)
    └── consumed_by: Stage 6, Stage 7

Stage 6 (Regime)
└── produces: Regime labels (9 composites)
    └── consumed_by: Stage 7

Stage 7 (Label Optimization)
├── produces:
│   ├── optimal_barrier_params.json
│   ├── Labeled DataFrame
│   └── label_optimization_history.csv
└── consumed_by: Stage 8, Stage 16

Stage 8 (Feature Selection)
├── produces:
│   ├── optimal_feature_mask.json
│   └── selected_features.txt (~60-100)
└── consumed_by: Stage 9, Stage 12, Stage 16

Stage 9 (Feature Pruning)
├── produces:
│   ├── pruned_feature_mask.json
│   ├── feature_importance_ranking.csv
│   └── Final feature set (~30-60)
└── consumed_by: Stage 10, Stage 12, Stage 16

Stage 10 (Splits)
└── produces: Train/Val/Test DataFrames
    └── consumed_by: Stage 11

Stage 11 (Scaling)
├── produces:
│   ├── data/splits/scaled/{symbol}_{split}.parquet
│   └── scaler.pkl
└── consumed_by: Stage 12, Stage 16

Stage 12 (Adaptation)
└── produces: Model-specific tensors (2D/3D/4D)
    └── consumed_by: Stage 13, Stage 14

Stage 13 (Hyperparameter Optimization)
├── produces:
│   ├── {model}_best_params.json (×23)
│   └── {model}_optimization_history.csv (×23)
└── consumed_by: Stage 14, Stage 16

Stage 14 (Training)
├── produces:
│   ├── Trained models (.pkl/.pt)
│   ├── Training reports
│   └── OOF predictions
└── consumed_by: Stage 15, Stage 16

Stage 15 (Stacking)
├── produces:
│   ├── Meta-learner models
│   └── Stacking performance report
└── consumed_by: Stage 16

Stage 16 (Bundling)
├── consumes_from:
│   ├── Stage 7: barrier_params.json
│   ├── Stage 8-9: feature_mask.json
│   ├── Stage 11: scaler.pkl
│   ├── Stage 13: hyperparameters.json
│   └── Stage 14-15: trained models
└── produces: ModelBundle V1.1.0


================================================================================
                    PROPOSED DIRECTORY STRUCTURE
================================================================================

src/
├── pipeline/                          # NEW - Unified Pipeline
│   ├── __init__.py
│   ├── unified.py                     # MLPipeline master orchestrator
│   ├── config.py                      # MLConfig unified configuration
│   ├── state.py                       # PipelineState management
│   └── phases/
│       ├── __init__.py
│       ├── data.py                    # Stages 1-6
│       ├── optimization.py            # Stages 7-9
│       ├── training.py                # Stages 10-15
│       └── deployment.py              # Stage 16
│
├── phase1/                            # KEEP - Data Pipeline (wrapped by phases/data.py)
│   └── stages/
│       ├── ingest/
│       ├── clean/
│       ├── sessions/
│       ├── mtf/
│       ├── features/
│       ├── regime/
│       ├── labeling/
│       ├── splits/
│       ├── scaling/
│       └── datasets/adapters/
│
├── models/                            # KEEP - Model implementations
│   ├── base.py
│   ├── registry.py
│   ├── trainer.py
│   ├── boosting/
│   ├── classical/
│   ├── neural/
│   └── ensemble/
│
├── optimization/                      # KEEP - Optuna optimization
│   └── search_spaces.py
│
├── cross_validation/                  # KEEP - CV infrastructure
│   ├── cpcv.py
│   ├── oof_io.py
│   ├── oof_stacking.py
│   └── walk_forward.py
│
└── cli/                               # NEW - Unified CLI
    └── unified_cli.py

config/
├── global.yaml
├── labeling.yaml
├── pipeline/
│   ├── training.yaml
│   └── cv.yaml
├── features/
│   ├── model_features.yaml
│   ├── mtf_strategies.yaml
│   └── selection_methods.yaml
├── optimization/
│   ├── label_optimization.yaml       # Stage 7
│   ├── feature_selection.yaml        # Stage 8
│   ├── feature_pruning.yaml          # Stage 9
│   └── hyperparameter.yaml           # Stage 13
├── models/
│   └── {23 model config files}
└── ensembles/
    ├── boosting_trio.yaml
    ├── temporal_stack.yaml
    └── meta_learner.yaml


================================================================================
                    CRITICAL FILES FOR IMPLEMENTATION
================================================================================

Priority 1 - Entry Points:
├── src/pipeline/unified.py            # Master MLPipeline orchestrator
├── src/pipeline/config.py             # MLConfig unified configuration
└── src/pipeline/state.py              # PipelineState management

Priority 2 - Phase Wrappers:
├── src/pipeline/phases/data.py        # Wraps Stages 1-6
├── src/pipeline/phases/optimization.py # Wraps Stages 7-9
├── src/pipeline/phases/training.py    # Wraps Stages 10-15
└── src/pipeline/phases/deployment.py  # Wraps Stage 16

Priority 3 - Already Complete:
├── src/models/base.py                 # BaseModel interface (EXISTS)
├── src/models/registry.py             # Model registry (EXISTS)
├── src/phase1/stages/datasets/adapters/multi_resolution.py  # 619 lines (EXISTS)
└── config/optimization/*.yaml         # All 4 optimization configs (EXISTS)

================================================================================
```
