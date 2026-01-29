# ML Factory - Complete Feature Inventory

**Generated:** 2026-01-29
**Codebase:** `/Users/sneh/research/src/` (460 Python files)

ML Factory is a config-driven system for building production ML ensembles for financial time-series prediction.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Core System](#core-system)
3. [Data Pipeline](#data-pipeline)
4. [Feature Engineering](#feature-engineering)
5. [Models](#models)
6. [Optimization](#optimization)
7. [Validation](#validation)
8. [Inference](#inference)
9. [Configuration](#configuration)
10. [CLI](#cli)
11. [Production Systems](#production-systems)

---

## System Overview

### At a Glance

| Metric | Value |
|--------|-------|
| Total Python Files | 460 |
| Registered Models | 23 |
| Feature Families | 15 |
| Total Features | 196 |
| Pipeline Stages | 12 |
| CV Methods | 4 |
| Labeling Strategies | 6 |
| Adapter Types | 4 |

### Core Flow

```
Raw OHLCV → Pipeline (12 stages) → Features + Labels → Adapters → Models → Ensemble
```

### Key Guarantees

| Guarantee | How |
|-----------|-----|
| No data leakage | Purge/embargo in all CV splits |
| No lookahead | All MTF operations use `shift(1)` |
| Reproducible | Same config = same output |
| Realistic metrics | Transaction costs, slippage included |

---

## Core System

**Location:** `src/core/`

### Types and Enums

**File:** `src/core/types.py`

| Enum | Values | Purpose |
|------|--------|---------|
| `DataRank` | `TABULAR_2D`, `SEQUENCE_3D`, `MULTI_TF_4D` | Tensor dimensionality |
| `ModelFamily` | `BOOSTING`, `CLASSICAL`, `NEURAL`, `ENSEMBLE`, `META_LEARNER`, `TRANSFORMER` | Model classification |
| `FeatureFamily` | 13 families (196 features) | Feature grouping |
| `TrainingMode` | `STANDARD`, `WALK_FORWARD`, `REGIME_AWARE`, `META_LABELING` | Training variants |
| `CVMethod` | `PURGED_KFOLD`, `CPCV`, `WALK_FORWARD`, `PBO` | Cross-validation |
| `AdapterType` | `TABULAR`, `SEQUENCE`, `MULTI_STREAM` | Data transformation |
| `LabelingMethod` | `TRIPLE_BARRIER`, `DIRECTIONAL`, `THRESHOLD`, `REGRESSION` | Target generation |

### Contract System

**Location:** `src/core/contracts/`

**DataContract** - Schema for data validation:
- `FeatureMode`: ENGINEERED, RAW, HYBRID, OOF_PROBS
- `MTFMode`: NONE, INDICATORS, MULTI_STREAM
- `DATA_SCHEMA`: Standard schema definition

**ModelContract** - 23 models registered with contracts specifying:
- Input rank (2D/3D/4D)
- MTF mode
- Scaling requirements
- Feature bounds (min/max)

**FeatureSpec** - 5-dimension specification:
1. Triple Barrier Parameters
2. Feature Selection
3. Feature Parameters
4. Feature Timeframes
5. Model Hyperparameters

### Exceptions

**File:** `src/core/exceptions.py`

| Exception | Purpose |
|-----------|---------|
| `DataContractViolation` | Schema validation failure |
| `LeakageDetected` | Data leakage found |
| `LookaheadBiasDetected` | Future data accessed |
| `ModelNotFoundError` | Unknown model requested |
| `ConfigurationError` | Invalid configuration |
| `TrainingError` | Training failure |
| `InferenceError` | Prediction failure |

---

## Data Pipeline

**Location:** `src/data/pipeline/`

### Pipeline Stages (12)

| # | Stage | Directory | Key Functions |
|---|-------|-----------|---------------|
| 1 | Data Ingestion | `ingest/` | `run_data_generation`, `DataIngestor` |
| 2 | Data Cleaning | `clean/` | `run_data_cleaning`, `DataCleaner` |
| 3 | Feature Engineering | `features/` | `run_feature_engineering`, `FeatureEngineer` |
| 4 | Initial Labeling | `labeling/` | `run_initial_labeling` |
| 5 | GA Optimization | `ga_optimize/` | `run_ga_optimization` |
| 6 | Final Labels + Quality | `final_labels/` | `run_final_labels` |
| 7 | Train/Val/Test Splits | `splits/` | `run_create_splits` |
| 7.5 | Feature Scaling | `scaling/` | `run_feature_scaling`, `FeatureScaler` |
| 7.6 | Dataset Building | `datasets/` | `run_build_datasets` |
| 8 | Data Validation | `validation/` | `run_validation` |
| 9 | Report Generation | `reporting/` | `run_generate_report` |
| 10 | Evaluation | `evaluation/` | `run_evaluation` |

### Additional Pipeline Modules

| Module | Purpose |
|--------|---------|
| `mtf/` | Multi-timeframe processing |
| `meta_labeling/` | Meta-labeling and bet sizing |
| `regime/` | Regime detection |
| `sessions/` | Trading session handling |
| `scaled_validation/` | Post-scaling validation |

### Labeling Strategies

**Location:** `src/data/pipeline/stages/labeling/`

| Strategy | Class | Description |
|----------|-------|-------------|
| Triple Barrier | `TripleBarrierLabeler` | Lopez de Prado with ATR-based barriers |
| Adaptive Triple Barrier | `AdaptiveTripleBarrierLabeler` | Regime-adaptive labeling |
| Directional | `DirectionalLabeler` | Simple direction of return |
| Threshold | `ThresholdLabeler` | Percentage threshold-based |
| Regression | `RegressionLabeler` | Continuous return targets |
| Meta-Labeling | `MetaLabeler` | Two-stage with bet sizing |

**Numba-optimized functions:**
- `triple_barrier_numba(...)` - Fast triple barrier
- `triple_barrier_numba_with_costs(...)` - With transaction costs

### Adapter Types

**Location:** `src/data/adapters/`

| Adapter | Output Shape | Target Models |
|---------|--------------|---------------|
| `TabularAdapter` | (n_samples, n_features) | Boosting, Classical, Meta-learners |
| `SequenceAdapter` | (n_samples, seq_len, n_features) | LSTM, GRU, TCN, TFT |
| `MultiStreamAdapter` | (n_samples, n_tfs, seq_len, n_features) | PatchTST, iTransformer |
| `MultiResolution4DAdapter` | (n_samples, n_timeframes, seq_len, features) | 4D Transformers |

**Supporting Classes:**
- `AdapterRegistry` - Registry for adapter types
- `AdapterFactory` - Config-driven creation
- `AdapterScaler` - Fit-on-train scaling
- `UnifiedDataPreparation` - Complete data prep pipeline
- `OOFAligner` - Alignment for heterogeneous ensembles

---

## Feature Engineering

**Location:** `src/data/features/`

### Feature Families (15 families, 196 features)

| Family | Count | Examples |
|--------|-------|----------|
| `raw` | 5 | open, high, low, close, volume |
| `momentum` | 23 | RSI (7,14,21), MACD, Stochastic, Williams %R, ROC, CCI, MFI |
| `moving_average` | 16 | SMA (10,20,50,200), EMA (10,20,50,200), crossovers |
| `volatility` | 25 | ATR, Bollinger Bands, Keltner, GARCH, Parkinson, Garman-Klass, Rogers-Satchell, Yang-Zhang |
| `volume` | 15 | OBV, VWAP, TWAP, dollar volume, volume ratios |
| `trend` | 6 | ADX, Supertrend |
| `price` | 12 | returns, price ratios, autocorrelation |
| `microstructure` | 15 | Amihud illiquidity, Roll spread, Kyle lambda, Corwin-Schultz |
| `entropy` | 12 | Shannon, Lempel-Ziv, Approximate entropy, Sample entropy, Hurst |
| `wavelets` | 15 | DWT coefficients, energy, trend strength, volatility |
| `temporal` | 9 | hour/day/month encoding, session flags |
| `regime` | 9 | volatility regime, trend regime |
| `order_flow` | 12 | buy/sell pressure, volume delta |
| `liquidity` | 12 | spread estimation, liquidity regime |
| `mean_reversion` | 10 | z-scores, Hurst, variance ratios |

### Key Feature Functions

```python
compute_all_features(df)                      # All 196 features
compute_features_by_family(df, families)      # By family
compute_features_by_names(df, feature_names)  # Specific features
compute_single_feature(df, feature_name)      # One feature
```

### MTF (Multi-Timeframe) Features

**File:** `src/data/features/compute/mtf.py`

- `MTFFeatureComputer` - Computes features across timeframes
- `compute_mtf_features(df, config)` - Main MTF computation
- `resample_ohlcv(df, timeframe)` - Resampling with `shift(1)` anti-lookahead

---

## Models

**Location:** `src/models/`

### Complete Model Registry (23 Models)

#### Boosting Models (3)

| Model | File | Key Parameters |
|-------|------|----------------|
| XGBoost | `boosting/xgboost_model.py` | depth, learning_rate, n_estimators, reg_alpha/lambda |
| LightGBM | `boosting/lightgbm_model.py` | num_leaves, learning_rate, n_estimators |
| CatBoost | `boosting/catboost_model.py` | depth, learning_rate, iterations, l2_leaf_reg |

#### Classical Models (3)

| Model | File | Key Parameters |
|-------|------|----------------|
| Random Forest | `classical/random_forest.py` | n_estimators, max_depth, min_samples_split |
| Logistic Regression | `classical/logistic.py` | C, penalty, solver |
| SVM | `classical/svm.py` | C, kernel, gamma |

#### Neural RNN Models (2)

| Model | File | Key Parameters |
|-------|------|----------------|
| LSTM | `neural/lstm_model.py` | hidden_size, n_layers, dropout, bidirectional |
| GRU | `neural/gru_model.py` | hidden_size, n_layers, dropout, bidirectional |

#### Neural CNN Models (3)

| Model | File | Key Parameters |
|-------|------|----------------|
| TCN | `neural/tcn_model.py` | n_filters, kernel_size, n_layers, dropout |
| InceptionTime | `neural/inceptiontime_model.py` | n_filters, depth, bottleneck_size |
| 1D ResNet | `neural/resnet1d_model.py` | n_filters, n_blocks, kernel_size |

#### Transformer Models (4)

| Model | File | Key Parameters |
|-------|------|----------------|
| Transformer | `neural/transformer_model.py` | d_model, n_heads, n_layers, dropout |
| PatchTST | `neural/patchtst_model.py` | d_model, n_heads, patch_length, stride |
| iTransformer | `neural/itransformer_model.py` | d_model, n_heads, n_layers |
| TFT | `neural/tft_model.py` | hidden_size, n_heads, attention_heads |

#### MLP Models (1)

| Model | File | Key Parameters |
|-------|------|----------------|
| N-BEATS | `neural/nbeats_model.py` | stack_types, n_blocks, width, sharing |

#### Meta-Learners (4)

| Model | File | Key Parameters |
|-------|------|----------------|
| Ridge Meta | `ensemble/ridge_meta.py` | alpha |
| MLP Meta | `ensemble/mlp_meta.py` | hidden_layers, dropout |
| Calibrated Meta | `ensemble/calibrated_meta.py` | method (isotonic/platt) |
| XGBoost Meta | `ensemble/xgboost_meta.py` | (same as XGBoost) |

#### Ensemble Methods (3)

| Method | Description |
|--------|-------------|
| Voting | Majority/weighted voting |
| Stacking | OOF-based meta-learning |
| Blending | Holdout-based meta-learning |

### Training Capabilities

**File:** `src/models/trainer.py`

| Feature | Description |
|---------|-------------|
| `Trainer` | Main training orchestration |
| `train_model(model_name, container, horizon)` | Convenience function |
| `evaluate_model(model, X, y)` | Model evaluation |
| `compute_classification_metrics(y_true, y_pred)` | Metrics computation |

**Advanced Training Services:**
- `ModelTrainer` - Full training pipeline
- `ParallelTraining` - Multi-model parallel training
- `RegimeTrainer` - Regime-aware training
- `OOFGenerator` - Out-of-fold generation

### Calibration

**File:** `src/models/calibration/calibrator.py`

| Method | Use Case |
|--------|----------|
| Isotonic regression | Boosting models |
| Sigmoid/Platt scaling | Linear models |
| Auto-selection | Based on sample count |

### Conformal Prediction

**File:** `src/models/calibration/conformal.py`

| Method | Description |
|--------|-------------|
| LAC | Least Ambiguous Class-conditional |
| APS | Adaptive Prediction Sets |
| Coverage | Configurable (e.g., 90%) |

### Ensemble Diversity

**File:** `src/models/ensemble/diversity.py`

| Metric | Purpose |
|--------|---------|
| Correlation | Pairwise prediction correlation |
| Q-statistic | Pairwise association |
| Disagreement | Prediction divergence |
| Double fault | Joint error rate |
| Entropy | Prediction distribution |
| KL divergence | Distribution similarity |

---

## Optimization

**Location:** `src/optimization/`

### 5-Dimension Optimization

**File:** `src/optimization/five_dimension_objective.py`

| Dimension | Parameters | Search Space |
|-----------|------------|--------------|
| 1. Triple Barrier | profit_threshold, loss_threshold, max_holding_bars | ATR multipliers |
| 2. Feature Selection | which features to include | Boolean per feature |
| 3. Feature Parameters | RSI period, ATR window, etc. | 0.5x to 2x defaults |
| 4. Feature Timeframes | which TF per feature | 5min, 15min, 30min, 60min |
| 5. Model Hyperparameters | model-specific params | `HYPERPARAMETER_SPACES` |

### Feature Selection Methods

| Method | Class | Description |
|--------|-------|-------------|
| Optuna Selection | `FeatureOptimizer` | Binary include/exclude |
| Walk-Forward | `WalkForwardFeatureSelector` | Rolling window |
| Purged Selection | `PurgedSelector` | Time-aware with purging |
| OHLCV Selection | `OHLCVSelector` | Raw feature selection |

### Hyperparameter Tuning

**File:** `src/optimization/hyperparameters.py`

- `HyperparameterOptimizer` - Optuna-based for all 23 models
- `HYPERPARAMETER_SPACES` - Complete search spaces
- `suggest_hyperparameters(trial, model_name)` - Per-model suggestions
- `get_default_hyperparameters(model_name)` - Defaults

---

## Validation

**Location:** `src/validation/`

### Leakage Detection

**File:** `src/validation/leakage_detection.py`

| Method | Description |
|--------|-------------|
| Feature-Label Correlation | Spearman/Pearson at point-in-time |
| Temporal Leakage | Forward vs backward correlation |
| Information Leakage | Mutual information analysis |
| Comprehensive | All three combined |

**Output:** `LeakageReport` with per-feature scores

### Lookahead Audit

**File:** `src/validation/lookahead_audit.py`

| Method | Description |
|--------|-------------|
| `audit_feature_lookahead` | Corruption-based testing |
| `audit_mtf_alignment` | MTF-specific check |
| `validate_resample_config` | Pandas resample validation |

**Methodology:**
1. Corrupt future data (NaN / random / shuffle)
2. Recompute features
3. Check if PAST feature values changed
4. If yes → LOOKAHEAD EXISTS

### Cross-Validation Methods

**File:** `src/validation/cv/`

| Method | Class | Description |
|--------|-------|-------------|
| Purged K-Fold | `PurgedKFold` | Time-series CV with label-aware purging |
| CPCV | `CombinatorialPurgedCV` | Combinatorial Purged CV |
| Walk-Forward | `WalkForwardEvaluator` | Expanding/sliding window |
| PBO | `compute_pbo()` | Probability of Backtest Overfitting |

**Configuration:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `purge_bars` | 60 | Remove overlapping labels |
| `embargo_bars` | 1440 | Buffer (~1 trading day) |
| `n_splits` | 5 | Number of folds |

### Statistical Testing

**File:** `src/validation/statistical_tests.py`

| Test | Purpose |
|------|---------|
| Diebold-Mariano | Forecast accuracy comparison |
| Paired t-test | Paired observations |
| Wilcoxon | Non-parametric alternative |

**Deflated Sharpe Ratio:**
- `compute_deflated_sharpe(sharpe, n_trials, returns)`
- `compute_dsr_from_optuna_study(study)`
- `dsr_gate(dsr, threshold)` - Deploy/reject decision

**Bootstrap Confidence Intervals:**
- BCa (bias-corrected accelerated) method
- Sharpe ratio, max drawdown, accuracy, F1, win rate

---

## Inference

**Location:** `src/inference/`

### Backtesting

**File:** `src/inference/backtesting/`

**Backtester:**
- `Backtester` - Main simulation
- `BacktestConfig` - With symbol presets (e.g., `for_mes()`)
- `BacktestResult` - Results container

**Transaction Costs:**
- `FixedSlippage`, `LinearSlippage`, `SquareRootSlippage`, `VolatilityScaledSlippage`
- `CostCalculator` - Unified calculation

**Position Sizing:**
| Method | Description |
|--------|-------------|
| `KellyCriterion` | Optimal growth-maximizing |
| `FixedFractional` | Risk fixed % per trade |
| `VolatilityTargeted` | Scale by inverse volatility |
| `EqualWeight` | Equal allocation |
| `FixedContracts` | Fixed position size |

**Performance Metrics:**
- Sharpe, Sortino, Calmar ratios
- Max drawdown, drawdown duration
- Win rate, profit factor, expectancy
- VaR, CVaR (tail risk)

### Prediction Methods

**File:** `src/inference/orchestrator.py`

| Method | Description |
|--------|-------------|
| `InferenceOrchestrator.from_experiment(config)` | Load from experiment |
| `InferenceOrchestrator.from_bundle(path)` | Load from bundle |
| `orchestrator.predict(X_new)` | Standard prediction |
| `orchestrator.predict_from_raw(raw_ohlcv_df)` | End-to-end from raw |
| `orchestrator.predict_batch(data, output_path)` | Batch inference |

**Model Bundles:**
- `ModelBundle` - Serializable container
- `PreprocessingGraph` - Serializable pipeline
- `BundleBuilder` - Create from training
- `EnsembleBundle` - For stacking ensembles

---

## Configuration

**Location:** `src/config/`

### Primary Interface

```python
from src.config import UnifiedConfig, get_unified_config, get_config_value
```

### Configuration Categories

| Category | Classes |
|----------|---------|
| **Data** | `FeatureConfig`, `LabelingConfig`, `ScalerConfig`, `SequenceConfig`, `MTFConfig`, `SplitConfig` |
| **Training** | `TrainerConfig`, `OptunaConfig`, `GAConfig`, `CalibrationConfig`, `ConformalConfig` |
| **CV** | `CVConfig`, `PurgedKFoldConfig`, `CPCVConfig`, `WalkForwardConfig`, `PBOConfig` |
| **Model** | `XGBoostConfig`, `LightGBMConfig`, `LSTMConfig`, `TransformerConfig`, etc. |
| **Ensemble** | `EnsembleConfig`, `MetaLearnerConfig`, `StackingConfig`, `VotingConfig` |
| **Inference** | `InferenceConfig`, `BacktestConfig`, `BundleConfig`, `ServerConfig` |

### SmartConfig API ("ML for Dummies")

```python
from src.config.smart_config import train, SmartConfig

# Simple training entry point
result = train(model="xgboost", data=df)
```

---

## CLI

**Location:** `src/cli/`

### Available Commands

| Command | Description |
|---------|-------------|
| `ml run` | Full pipeline (data + training + evaluation) |
| `ml data` | Data pipeline only |
| `ml train model` | Train model(s) |
| `ml train ensemble` | Train ensemble |
| `ml cv` | Run cross-validation |
| `ml walk-forward` | Run walk-forward evaluation |
| `ml cpcv-pbo` | Run CPCV/PBO evaluation |
| `ml status` | Show pipeline status |
| `ml resume` | Resume from checkpoint |
| `ml version` | Show version information |

### Usage

```bash
python -m src.cli --help
python -m src.cli run --help
python -m src.cli train model --help
```

---

## Production Systems

### Label Quality and Sample Weighting

**File:** `src/data/pipeline/stages/final_labels/core.py`

**5 Quality Metrics:**
1. Speed Score - Normalized bars to hit
2. MAE Score - Max Adverse Excursion
3. MFE Score - Max Favorable Excursion
4. Pain-to-Gain Ratio - Risk per unit profit
5. Time-Weighted Drawdown - Penalizes long drawdowns

**Sample Weight Tiers:**
| Tier | Percentile | Weight |
|------|------------|--------|
| 1 | Top 20% | 1.5 |
| 2 | Middle 60% | 1.0 |
| 3 | Bottom 20% | 0.5 |

### Meta-Labeling and Bet Sizing

**Location:** `src/data/pipeline/stages/meta_labeling/`

**Two-Stage System:**
1. Primary Model (RECALL-optimized) - Capture all potential trades
2. Meta-Model (PRECISION-optimized) - Filter by confidence

**Bet Sizing:** Concave function from meta_proba to position size

### Regime Detection

**Files:** `src/models/training/regime_detector.py`, `src/data/pipeline/stages/regime/`

| Method | States |
|--------|--------|
| `volatility_percentile` | low / medium / high |
| `trend_adx` | downtrend / sideways / uptrend |
| `combined` | 9 states (3x3 grid) |
| `HMM` | Unsupervised |

### Drift Monitoring

**Location:** `src/validation/monitoring/`

| Detector | Method |
|----------|--------|
| ADWIN | Adaptive Windowing |
| PSI | Population Stability Index |
| KS | Kolmogorov-Smirnov |

**Alert System:**
- `AlertHandler` - Severity levels, rate limiting
- `DriftAlertAggregator` - Time-windowed summarization

### Experiment Tracking

**File:** `src/models/tracking/mlflow_tracker.py`

| Tracker | Description |
|---------|-------------|
| `MLflowTracker` | Full MLOps tracking |
| `LocalTracker` | File-based fallback |

**Logs:** parameters, metrics, artifacts, models, tags

### Financial Reporting

**File:** `src/models/evaluation/financial_report.py`

**Contents:** performance metrics, trade statistics, classification metrics, direction-specific analysis

**Formats:** HTML, JSON, Markdown

---

## Summary

ML Factory provides a complete, production-ready system for:

1. **Data Processing** - 12-stage pipeline with validation at each step
2. **Feature Engineering** - 196 features across 15 families with MTF support
3. **Modeling** - 23 models from boosting to transformers
4. **Optimization** - 5-dimension Optuna optimization with CPCV
5. **Validation** - Leakage detection, lookahead audit, purged CV
6. **Inference** - Backtesting with realistic costs, model bundles
7. **Production** - Regime detection, drift monitoring, meta-labeling

All with config-driven design, no data leakage guarantees, and comprehensive documentation.

---

*See TECHNICAL_IMPROVEMENTS.md for optimization opportunities and IMPROVEMENTS.md for ML/trading improvements.*
