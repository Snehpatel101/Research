# ML Factory: Data Flow Reference

**Version:** 1.1
**Purpose:** Complete data transformation guide for AI agent ingestion
**Scope:** How data flows through the pipeline

---

## 0. THE SINGLE ENTRY POINT

```python
from src import MLFactory, PipelineConfig

# Configure the pipeline
config = PipelineConfig(
    symbol="MES",
    data_path="./data/mes.parquet",
    output_dir="./experiments",
    models=["xgboost", "lightgbm", "lstm"],
    training_mode="standard",  # or "regime_aware", "meta_labeling"
    build_ensemble=True,
    compute_mtf_features=True,
    optimize_labels=True,
)

# Run entire pipeline
factory = MLFactory(config)
result = factory.run(df)

# Access results
print(f"Best model: {result.best_model}")
bundle = result.get_inference_bundle()
predictions = bundle.predict(new_data)
```

**Key Files:**
- `src/factory.py` - MLFactory class
- `src/core/config.py` - PipelineConfig
- `src/core/constants.py` - All defaults
- `src/core/interfaces.py` - All contracts

---

## 1. End-to-End Data Flow

```
RAW OHLCV (1-min bars)
         |
         v
+------------------+
| 1. INGESTION     |   Load parquet/CSV, validate schema
+------------------+
         |
         v
+------------------+
| 2. CLEANING      |   Fill gaps, remove outliers, session filter
+------------------+
         |
         v
+------------------+
| 3. MTF RESAMPLE  |   1m -> 5m, 10m, 15m, 20m, 25m, 30m, 45m, 1h
+------------------+
         |
         v
+------------------+
| 4. FEATURES      |   162 indicators per timeframe (12 families)
+------------------+
         |
         v
+------------------+
| 5. REGIME        |   Volatility/trend classification
+------------------+
         |
         v
+========================+
| 6. LABEL OPTIM   |   Optuna: 100 trials (barrier params)
| [OPTUNA STAGE 1] |   upper_mult, lower_mult, horizon, atr_period
+========================+
         |
         v
+========================+
| 7. FEATURE SELECT|   Optuna: 100 trials (binary selection)
| [OPTUNA STAGE 2] |   Include/exclude per feature
+========================+
         |
         v
+========================+
| 8. FEATURE PRUNE |   Optuna: 50 trials (importance-based)
| [OPTUNA STAGE 3] |   Remove low-value features
+========================+
         |
         v
+------------------+
| 9. SPLITTING     |   Train 70% | Val 15% | Test 15%
+------------------+
         |
         v
+------------------+
| 10. SCALING      |   RobustScaler fit on train only
+------------------+
         |
         v
+------------------+
| 11. ADAPTATION   |   2D/3D/4D tensor per model type
+------------------+
         |
         v
+========================+
| 12. HYPERPARAM   |   Optuna: 100 trials/model
| [OPTUNA STAGE 4] |   Per-model search spaces (23 models)
+========================+
         |
         v
+------------------+
| 13. TRAINING     |   PurgedKFold CV, OOF generation
+------------------+
         |
         v
+------------------+
| 14. STACKING     |   OOF alignment, meta-learner
+------------------+
         |
         v
+------------------+
| 15. BUNDLING     |   Model + Scaler + Graph -> Artifact
+------------------+
         |
         v
+------------------+
| 16. INFERENCE    |   Raw OHLCV -> Prediction
+------------------+
```

**Optuna Optimization Stages (Total: 100 + 100 + 50 + 100×N trials)**
- Stage 6: Label optimization (triple-barrier parameters)
- Stage 7: Feature selection (binary include/exclude)
- Stage 8: Feature pruning (importance-based removal)
- Stage 12: Hyperparameter optimization (per-model, all 23 models)

---

## 2. Stage Details

### 2.1 Ingestion

**Input:** Raw parquet/CSV files
**Output:** Validated DataFrame with DatetimeIndex

```
Required columns: [datetime, open, high, low, close, volume]
Index: pd.DatetimeIndex (UTC)
Frequency: 1 minute (canonical)
```

### 2.2 Cleaning

**Input:** Raw OHLCV DataFrame
**Output:** Clean OHLCV DataFrame

```
Operations:
1. Fill small gaps (< 5 min) with forward fill
2. Remove outliers (> 5 ATR from median)
3. Filter to trading sessions only
4. Verify no future data leakage
```

### 2.3 MTF Resampling

**Input:** 1-min OHLCV
**Output:** 9 timeframe parquet files

```
Timeframes: [1m, 5m, 10m, 15m, 20m, 25m, 30m, 45m, 1h]

Aggregation:
- open: first
- high: max
- low: min
- close: last
- volume: sum
```

### 2.4 Feature Engineering

**Input:** OHLCV per timeframe
**Output:** 162 base feature columns (see PHASE_1_UNIFIED_FEATURES.md)

```
12 Feature Families (162 base features total):
1. Momentum (23): RSI, MACD, Stochastic, Williams, ROC, CCI, MFI
2. Moving Avg (16): SMA, EMA, crossovers, ratios
3. Volatility (25): ATR, BB, Keltner, HV, Parkinson, GK, GARCH
4. Volume (15): OBV, VWAP, TWAP, dollar volume
5. Trend (6): ADX, Supertrend
6. Price (12): Returns, ratios, autocorrelation, CLV
7. Microstructure (15): Amihud, Roll, Kyle Lambda
8. Entropy (12): Shannon, LZ, ApEn, SampEn, Hurst
9. Wavelets (15): DWT coefficients, energy
10. Temporal (9): Hour/day encoding, session progress, session flags
11. Regime (9): Vol/trend regime flags
12. MTF (~30/TF): Higher TF indicators with shift(1)

Total: 162 base + ~240 MTF = ~402 features available
```

### 2.5 Regime Detection

**Input:** Feature DataFrame
**Output:** Regime columns added

```
Volatility Regimes (ATR percentile):
- low_vol: ATR < 25th percentile
- normal_vol: 25th <= ATR <= 75th
- high_vol: ATR > 75th percentile

Trend Regimes (ADX):
- trending: ADX > 25
- sideways: ADX <= 25

Composite: 6 combinations (low_vol_trending, etc.)
```

### 2.6 Labeling

**Input:** Feature DataFrame + OHLCV
**Output:** Label column (-1, 0, +1)

```
Triple-Barrier Method:
- Upper barrier: +ATR_multiplier * ATR
- Lower barrier: -ATR_multiplier * ATR
- Timeout: horizon bars

Optuna Optimization:
- Objective: class balance + predictability
- Trials: 50-100
- Parameters: ATR_mult, horizon
```

### 2.7 Splitting

**Input:** Labeled DataFrame
**Output:** Train/Val/Test DataFrames

```
Split ratios: 70% / 15% / 15%
Method: Chronological (no shuffle)
Purge gap: 60 bars between splits
Embargo: 1440 bars after test start
```

### 2.8 Scaling

**Input:** Split DataFrames
**Output:** Scaled DataFrames

```
Scaler: RobustScaler (median, IQR)
Fit: Train set only
Transform: All sets
Clip: [-5, 5] after scaling
Save: scaler.pkl for inference
```

### 2.9 Data Adaptation

**Input:** Scaled DataFrame
**Output:** Model-ready tensors

```
TabularAdapter (2D):
  Input: DataFrame (N, F)
  Output: numpy array (N, F)
  Models: XGBoost, LightGBM, RF, Logistic, SVM

SequenceAdapter (3D):
  Input: DataFrame (N, F)
  Output: numpy array (N-S+1, S, F)
  Models: LSTM, GRU, TCN, Transformer
  Note: Loses first S-1 samples

MultiStreamAdapter (4D):
  Input: Dict[TF -> DataFrame]
  Output: numpy array (N', T, S, F)
  Models: PatchTST, iTransformer
```

### 2.10 Training

**Input:** Adapted tensors
**Output:** Trained models + OOF predictions

```
CV: PurgedKFold (5 folds boosting, 3 folds neural)
Purge: 60 bars before test fold
Embargo: 1440 bars after test fold

OOF Generation:
- Train on K-1 folds
- Predict on held-out fold
- Stack all predictions
```

### 2.11 Stacking

**Input:** OOF predictions from all models
**Output:** Stacking dataset + meta-learner

```
Alignment (heterogeneous models):
- Tabular: 100% coverage, offset=0
- Sequence: ~98% coverage, offset=seq_len-1
- Common samples: N - max_offset

Stacking Features:
- {model}_prob_short
- {model}_prob_neutral
- {model}_prob_long
- mean_confidence
- prediction_agreement
- prediction_entropy
```

### 2.12 Bundling

**Input:** Trained model + scaler + config
**Output:** Serialized bundle directory

```
Bundle Structure:
bundles/xgb_h20/
  manifest.json      # File list + checksums
  metadata.json      # Model info, metrics
  features.json      # Ordered column names
  scaler.pkl         # Fitted scaler
  preprocessing_graph.json  # Full pipeline config
  model/             # Model artifacts
```

### 2.13 Inference

**Input:** Raw OHLCV or pre-computed features
**Output:** Predictions (-1, 0, +1) + probabilities

```
Path A (Pre-computed features):
  features -> scaler.transform() -> model.predict()

Path B (Raw OHLCV):
  raw -> PreprocessingGraph.transform() -> model.predict()

Output:
  PredictionOutput(
    class_predictions: [-1, 0, 1]
    class_probabilities: [[p_short, p_neutral, p_long], ...]
    confidence: [max(probs), ...]
  )
```

---

## 3. Data Shape Transformations

```
Stage             Shape                  Example (N=100k, F=95, S=60)
-----             -----                  ----------------------------
Raw OHLCV         (N, 5)                 (100000, 5)
After Features    (N, F)                 (100000, 95)
Train Split       (0.7*N, F)             (70000, 95)

TabularAdapter    (N, F)                 (70000, 95)
SequenceAdapter   (N-S+1, S, F)          (69941, 60, 95)
MultiStream       (N', T, S, F)          (69941, 3, 60, 5)

OOF (Tabular)     (N, 3)                 (70000, 3)
OOF (Sequence)    (N-S+1, 3)             (69941, 3)
Aligned OOF       (common_N, 3*M)        (69941, 9) for 3 models

Stacking          (common_N, meta_F)     (69941, 12)
Meta-learner      (common_N, 3)          (69941, 3)
```

---

## 4. Anti-Leakage Safeguards

### 4.1 Temporal Leakage
- **Chronological splits only** - No random shuffling
- **Purge gaps** - 60 bars removed before test
- **Embargo period** - 1440 bars after test start

### 4.2 Feature Leakage
- **Shift(1) on MTF** - Higher TF features use previous bar
- **Train-only scaling** - Scaler fit on train only
- **No future labels** - Triple-barrier timeout verified

### 4.3 Validation
- **LookaheadAuditor** - Scans for future data usage
- **Leakage check** - Feature-label correlation analysis
- **PBO analysis** - Probability of backtest overfitting

---

## 5. Key Constants

```python
# Timeframes
CANONICAL_TIMEFRAMES = ["1m", "5m", "10m", "15m", "20m", "25m", "30m", "45m", "1h"]

# Splits
DEFAULT_SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}

# CV
DEFAULT_PURGE_BARS = 60
DEFAULT_EMBARGO_BARS = 1440

# Horizons
DEFAULT_HORIZONS = [5, 10, 15, 20]

# Sequence
DEFAULT_SEQUENCE_LENGTH = 60
```

---

## 6. Document Metadata

| Field | Value |
|-------|-------|
| Version | 1.1 |
| Created | 2026-01-17 |
| Updated | 2026-01-18 |
| Purpose | Data flow reference for AI agents |
| Related Docs | HIGH_LEVEL_ARCHITECTURE.md, PHASE_0-5.md |
| Entry Point | `from src import MLFactory, PipelineConfig` |
