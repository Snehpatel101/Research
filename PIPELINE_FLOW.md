# ML Model Factory - Pipeline Flow Map

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                        FUTURES TRADING SIGNAL FACTORY                                   │
│                     "Raw Bars → Trained Model → Trade Signals"                          │
└─────────────────────────────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════════════════════
                                    PHASE 1: DATA PIPELINE
                              (src/phase1/stages/ - 14 stages)
═══════════════════════════════════════════════════════════════════════════════════════════

┌──────────────────┐
│   RAW INPUT      │
│                  │
│  data/raw/       │
│  MES_1m.parquet  │     1-minute OHLCV bars
│  or .csv         │     (Open, High, Low, Close, Volume)
│                  │
│  Single contract │
│  per run         │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐     ┌─────────────────────────────────────────────────────────────┐
│  1. INGEST       │────▶│  • Validate OHLCV schema (timestamp, O, H, L, C, V)         │
│  loaders.py      │     │  • Check for gaps, duplicates, invalid prices              │
│  validators.py   │     │  • Convert timezone to UTC                                  │
└────────┬─────────┘     └─────────────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────┐     ┌─────────────────────────────────────────────────────────────┐
│  2. CLEAN        │────▶│  • Resample 1min → 5min bars                                │
│  cleaner.py      │     │  • Handle gaps (forward fill, interpolate, or drop)        │
│  gap_handler.py  │     │  • Remove after-hours / low-liquidity periods              │
│  bar_builders/   │     │  • Optional: dollar bars, volume bars, tick bars           │
└────────┬─────────┘     └─────────────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────┐     ┌─────────────────────────────────────────────────────────────┐
│  3. SESSIONS     │────▶│  • Filter to trading sessions (RTH or ETH)                  │
│  calendar.py     │     │  • Normalize session boundaries                             │
│  filter.py       │     │  • Handle holidays, half-days                               │
└────────┬─────────┘     └─────────────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────┐     ┌─────────────────────────────────────────────────────────────┐
│  4. FEATURES     │────▶│  150+ Technical Indicators:                                 │
│  engineer.py     │     │                                                             │
│                  │     │  PRICE:        returns, log_returns, range, gaps            │
│  momentum.py     │     │  MOMENTUM:     RSI, MACD, ROC, Williams %R, Stochastic     │
│  volatility.py   │     │  VOLATILITY:   ATR, Bollinger, Keltner, Parkinson, GK      │
│  trend.py        │     │  TREND:        ADX, Aroon, CCI, Ichimoku, SuperTrend       │
│  volume.py       │     │  VOLUME:       OBV, VWAP, MFI, AD line, volume profile     │
│  moving_avg.py   │     │  MA:           SMA, EMA, WMA, DEMA, TEMA, KAMA, HMA        │
│  wavelets.py     │     │  WAVELETS:     DWT decomposition (trend/noise separation)  │
│  microstructure  │     │  MICROSTRUC:   bid-ask proxy, order flow imbalance         │
│  temporal.py     │     │  TEMPORAL:     hour, day_of_week, session_progress         │
└────────┬─────────┘     └─────────────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────┐     ┌─────────────────────────────────────────────────────────────┐
│  5. REGIME       │────▶│  Market Regime Detection:                                   │
│  hmm.py          │     │                                                             │
│  volatility.py   │     │  • HMM-based regime (2-4 hidden states)                    │
│  trend.py        │     │  • Volatility regime (low/normal/high)                     │
│  composite.py    │     │  • Trend regime (trending/ranging/reversal)                │
│                  │     │  • Composite regime score                                   │
└────────┬─────────┘     └─────────────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────┐     ┌─────────────────────────────────────────────────────────────┐
│  6. MTF          │────▶│  Multi-Timeframe Features:                                  │
│  generator.py    │     │                                                             │
│                  │     │  • Resample features to 15min, 1H, 4H, Daily               │
│                  │     │  • Higher TF trend alignment                                │
│                  │     │  • Cross-timeframe divergences                              │
└────────┬─────────┘     └─────────────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                              LABELING SUBSYSTEM                                       │
│                        (The "Y" - What we're predicting)                             │
└──────────────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────┐     ┌─────────────────────────────────────────────────────────────┐
│  7. LABELING     │────▶│  Triple-Barrier Method (De Prado):                          │
│  triple_barrier  │     │                                                             │
│                  │     │     Upper Barrier (Take Profit)                             │
│                  │     │            ┬─────────────────────┐                          │
│                  │     │            │                     │  +1 LONG                 │
│                  │     │    Price ──┼─────────────────────┼──────▶                   │
│                  │     │            │                     │  -1 SHORT                │
│                  │     │            ┴─────────────────────┘                          │
│                  │     │     Lower Barrier (Stop Loss)                               │
│                  │     │                                                             │
│                  │     │  • If price hits upper first → +1 (LONG signal was correct)│
│                  │     │  • If price hits lower first → -1 (SHORT signal was correct)│
│                  │     │  • If neither within horizon → 0 (HOLD, no clear signal)   │
│                  │     │                                                             │
│                  │     │  Horizons: 5, 10, 15, 20 bars ahead                        │
│                  │     │  Barriers: ATR-based, symbol-specific (MES: 1.5:1.0 ratio) │
└────────┬─────────┘     └─────────────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────┐     ┌─────────────────────────────────────────────────────────────┐
│  8. GA_OPTIMIZE  │────▶│  Optuna Parameter Optimization:                             │
│  optuna_opt.py   │     │                                                             │
│  fitness.py      │     │  • Optimize barrier heights for each symbol                │
│                  │     │  • Objective: maximize Sharpe after transaction costs      │
│                  │     │  • Penalize class imbalance (want ~balanced labels)        │
│                  │     │  • Output: optimal (upper_mult, lower_mult, horizon)       │
└────────┬─────────┘     └─────────────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────┐     ┌─────────────────────────────────────────────────────────────┐
│  9. FINAL_LABELS │────▶│  Apply Optimized Parameters:                                │
│  core.py         │     │                                                             │
│                  │     │  • Re-run triple barrier with tuned params                 │
│                  │     │  • Generate quality scores per sample                       │
│                  │     │  • Mark label_end_time for each sample (for purging)       │
└────────┬─────────┘     └─────────────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                           SPLIT & SCALE SUBSYSTEM                                     │
│                     (Prevent Lookahead Bias - CRITICAL)                              │
└──────────────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────┐     ┌─────────────────────────────────────────────────────────────┐
│  10. SPLITS      │────▶│  Time-Series Aware Splitting:                               │
│  core.py         │     │                                                             │
│                  │     │  ┌─────────────────┬────────┬────────┐                      │
│                  │     │  │     TRAIN       │  VAL   │  TEST  │                      │
│                  │     │  │      70%        │  15%   │  15%   │                      │
│                  │     │  └─────────────────┴────────┴────────┘                      │
│                  │     │         TIME ──────────────────────────▶                    │
│                  │     │                                                             │
│                  │     │  NO SHUFFLING - strictly chronological                      │
│                  │     │                                                             │
│                  │     │  PURGE: 60 bars removed between splits                     │
│                  │     │         (prevents label overlap leakage)                    │
│                  │     │                                                             │
│                  │     │  EMBARGO: 1440 bars (~5 days) after each test fold         │
│                  │     │           (prevents serial correlation leakage)             │
└────────┬─────────┘     └─────────────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────┐     ┌─────────────────────────────────────────────────────────────┐
│  11. SCALING     │────▶│  Train-Only Robust Scaling:                                 │
│  scaler.py       │     │                                                             │
│                  │     │  • Fit scalers ONLY on training data                       │
│                  │     │  • Transform val/test with train-fitted scalers            │
│                  │     │  • RobustScaler (median/IQR) - outlier resistant           │
│                  │     │  • Clip extreme values to ±5 std                           │
│                  │     │                                                             │
│                  │     │  ⚠️  CRITICAL: Never fit on val/test = data leakage       │
└────────┬─────────┘     └─────────────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────┐     ┌─────────────────────────────────────────────────────────────┐
│  12. DATASETS    │────▶│  Build TimeSeriesDataContainer:                             │
│  container.py    │     │                                                             │
│  sequences.py    │     │  Unified interface for all model types:                    │
│                  │     │                                                             │
│                  │     │  TABULAR (XGBoost, LightGBM, RF, SVM, Logistic):           │
│                  │     │  ┌─────────────────────────────────────┐                    │
│                  │     │  │  X: (n_samples, n_features)         │  2D arrays        │
│                  │     │  │  y: (n_samples,)                    │                    │
│                  │     │  └─────────────────────────────────────┘                    │
│                  │     │                                                             │
│                  │     │  SEQUENCE (LSTM, GRU, TCN, Transformer):                   │
│                  │     │  ┌─────────────────────────────────────┐                    │
│                  │     │  │  X: (n_samples, seq_len, n_features)│  3D tensors       │
│                  │     │  │  y: (n_samples,)                    │  (lookback=60)    │
│                  │     │  └─────────────────────────────────────┘                    │
│                  │     │                                                             │
│                  │     │  Also includes: sample_weights, label_end_times            │
└────────┬─────────┘     └─────────────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────┐     ┌─────────────────────────────────────────────────────────────┐
│  13. VALIDATION  │────▶│  Data Quality Checks:                                       │
│  features.py     │     │                                                             │
│  integrity.py    │     │  • Feature correlation (drop if corr > 0.95)               │
│  drift.py        │     │  • Train/val distribution drift detection                  │
│                  │     │  • Label balance verification                               │
│                  │     │  • NaN/Inf checks                                           │
└────────┬─────────┘     └─────────────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────┐     ┌─────────────────────────────────────────────────────────────┐
│  14. REPORTING   │────▶│  Generate Pipeline Report:                                  │
│  charts.py       │     │                                                             │
│  sections.py     │     │  • Feature importance rankings                              │
│                  │     │  • Label distribution plots                                 │
│                  │     │  • Data quality summary                                     │
│                  │     │  • Ready-for-training confirmation                          │
└────────┬─────────┘     └─────────────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                              PHASE 1 OUTPUT                                           │
│                                                                                       │
│   TimeSeriesDataContainer {                                                          │
│       X_train: np.ndarray,     # (N_train, 150+ features)                           │
│       y_train: np.ndarray,     # Labels: -1, 0, +1                                  │
│       X_val:   np.ndarray,     # Validation set                                     │
│       y_val:   np.ndarray,                                                          │
│       X_test:  np.ndarray,     # Hold-out test (touch ONCE at very end)            │
│       y_test:  np.ndarray,                                                          │
│       sample_weights: np.ndarray,  # Quality-based (0.5x - 1.5x)                   │
│       label_end_times: pd.Series,  # For CV purging                                │
│       feature_names: List[str],                                                      │
│       horizon: int,            # 5, 10, 15, or 20 bars                              │
│   }                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════════════════════════════
                                 PHASE 2: MODEL TRAINING
                                (src/models/ - 13 models)
═══════════════════════════════════════════════════════════════════════════════════════════

                    ┌──────────────────────────────────────┐
                    │      TimeSeriesDataContainer         │
                    │         (from Phase 1)               │
                    └─────────────────┬────────────────────┘
                                      │
                                      ▼
                    ┌──────────────────────────────────────┐
                    │          MODEL REGISTRY              │
                    │       (Plugin Architecture)          │
                    │                                      │
                    │   @register(name="xgboost",          │
                    │            family="boosting")        │
                    │   class XGBoostModel(BaseModel):     │
                    │       def fit(...) → TrainingMetrics │
                    │       def predict(...) → Predictions │
                    │       def save/load(...)             │
                    └─────────────────┬────────────────────┘
                                      │
          ┌───────────────────────────┼───────────────────────────┐
          │                           │                           │
          ▼                           ▼                           ▼
┌─────────────────────┐   ┌─────────────────────┐   ┌─────────────────────┐
│  TABULAR MODELS     │   │  SEQUENCE MODELS    │   │  ENSEMBLE MODELS    │
│  (2D input)         │   │  (3D input)         │   │  (meta-learners)    │
│                     │   │                     │   │                     │
│  ├── XGBoost        │   │  ├── LSTM           │   │  ├── Voting         │
│  ├── LightGBM       │   │  ├── GRU            │   │  ├── Stacking       │
│  ├── CatBoost       │   │  ├── TCN            │   │  └── Blending       │
│  ├── Random Forest  │   │  └── Transformer    │   │                     │
│  ├── Logistic Reg   │   │                     │   │  ⚠️ Same-family     │
│  └── SVM            │   │  seq_len=60 bars    │   │     base models     │
│                     │   │  (lookback window)  │   │     only!           │
└─────────────────────┘   └─────────────────────┘   └─────────────────────┘
          │                           │                           │
          └───────────────────────────┴───────────────────────────┘
                                      │
                                      ▼
                    ┌──────────────────────────────────────┐
                    │            TRAINER                   │
                    │         (trainer.py)                 │
                    │                                      │
                    │  1. Prepare data (tabular or seq)   │
                    │  2. Initialize model from registry  │
                    │  3. Train with early stopping       │
                    │  4. Evaluate on validation          │
                    │  5. Save artifacts                  │
                    └─────────────────┬────────────────────┘
                                      │
                                      ▼
                    ┌──────────────────────────────────────┐
                    │         TRAINING OUTPUT              │
                    │                                      │
                    │  experiments/runs/{run_id}/          │
                    │  ├── model.pkl (or .pt for neural)  │
                    │  ├── training_metrics.json          │
                    │  ├── predictions_val.csv            │
                    │  ├── feature_importance.csv         │
                    │  └── config.yaml                    │
                    └──────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════════════════════════════
                            PHASE 3: CROSS-VALIDATION
                          (src/cross_validation/ - robust CV)
═══════════════════════════════════════════════════════════════════════════════════════════

┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                              PURGED K-FOLD CV                                            │
│                                                                                          │
│   Standard K-Fold is WRONG for time series - causes lookahead bias!                     │
│                                                                                          │
│   ❌ WRONG (standard CV):                                                                │
│   ┌─────┬─────┬─────┬─────┬─────┐                                                       │
│   │ F1  │ F2  │ F3  │ F4  │ F5  │  ← Random shuffling leaks future into past           │
│   └─────┴─────┴─────┴─────┴─────┘                                                       │
│                                                                                          │
│   ✅ CORRECT (Purged K-Fold):                                                            │
│                                                                                          │
│   Fold 1: [TRAIN TRAIN TRAIN]─purge─[TEST]                                              │
│   Fold 2: [TRAIN TRAIN]─purge─[TEST]─embargo─[TRAIN]                                    │
│   Fold 3: [TRAIN]─purge─[TEST]─embargo─[TRAIN TRAIN]                                    │
│   Fold 4: [TEST]─embargo─[TRAIN TRAIN TRAIN]                                            │
│                                                                                          │
│   PURGE:   Remove 60 bars around test fold (label overlap prevention)                   │
│   EMBARGO: Skip 1440 bars after test fold (serial correlation prevention)               │
│                                                                                          │
│   This ensures: Model NEVER sees future data during training                            │
└──────────────────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                           OUT-OF-FOLD (OOF) PREDICTIONS                                  │
│                                                                                          │
│   For each sample, prediction comes from model that NEVER saw it:                       │
│                                                                                          │
│   Sample 1000: Predicted by Fold 3 model (trained on Folds 1,2,4,5)                    │
│   Sample 2000: Predicted by Fold 1 model (trained on Folds 2,3,4,5)                    │
│                                                                                          │
│   OOF predictions are used for:                                                          │
│   • Unbiased model comparison                                                            │
│   • Stacking ensemble training (meta-learner input)                                     │
│   • PBO (Probability of Backtest Overfitting) calculation                               │
└──────────────────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                              WALK-FORWARD VALIDATION                                     │
│                                                                                          │
│   Simulates real trading: train on past, test on future, roll forward                   │
│                                                                                          │
│   Window 1: [═══TRAIN═══]─[TEST]                                                        │
│   Window 2:      [═══TRAIN═══]─[TEST]                                                   │
│   Window 3:           [═══TRAIN═══]─[TEST]                                              │
│   Window 4:                [═══TRAIN═══]─[TEST]                                         │
│                                                                                          │
│   • Expanding or sliding window options                                                  │
│   • Tests model stability over time                                                      │
│   • Detects regime changes                                                               │
└──────────────────────────────────────────────────────────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════════════════════════════
                              INFERENCE / DEPLOYMENT
                              (src/inference/ - production)
═══════════════════════════════════════════════════════════════════════════════════════════

┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                          │
│    NEW BAR ARRIVES                                                                       │
│         │                                                                                │
│         ▼                                                                                │
│   ┌───────────────┐     ┌───────────────┐     ┌───────────────┐                         │
│   │  Feature      │     │    Model      │     │   Trading     │                         │
│   │  Engineering  │────▶│   Predict     │────▶│   Decision    │                         │
│   │  (same as     │     │               │     │               │                         │
│   │   training)   │     │  P(long)=0.7  │     │   → LONG      │                         │
│   │               │     │  P(hold)=0.2  │     │   confidence  │                         │
│   │  150+ features│     │  P(short)=0.1 │     │   = 70%       │                         │
│   └───────────────┘     └───────────────┘     └───────────────┘                         │
│                                                                                          │
│   Output: { signal: +1, confidence: 0.70, probabilities: [0.1, 0.2, 0.7] }             │
│                                                                                          │
└──────────────────────────────────────────────────────────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════════════════════════════
                                  KEY PARAMETERS
═══════════════════════════════════════════════════════════════════════════════════════════

┌─────────────────────┬────────────────────────────────────────────────────────────────────┐
│ Parameter           │ Value / Description                                                │
├─────────────────────┼────────────────────────────────────────────────────────────────────┤
│ Symbol              │ MES, MGC, ES, GC (one per run, complete isolation)                │
│ Raw Timeframe       │ 1-minute bars                                                      │
│ Model Timeframe     │ 5-minute bars (after resampling)                                  │
│ Label Horizons      │ 5, 10, 15, 20 bars ahead                                          │
│ Train/Val/Test      │ 70% / 15% / 15%                                                   │
│ Purge Bars          │ 60 (= max_horizon × 3)                                            │
│ Embargo Bars        │ 1440 (~5 trading days at 5-min)                                   │
│ Sequence Length     │ 60 bars (for LSTM/GRU/TCN/Transformer)                            │
│ Features            │ 150+ indicators                                                    │
│ Classes             │ 3 (-1=SHORT, 0=HOLD, +1=LONG)                                     │
│ Sample Weights      │ 0.5x - 1.5x based on label quality                                │
└─────────────────────┴────────────────────────────────────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════════════════════════════
                                 ANTI-PATTERNS PREVENTED
═══════════════════════════════════════════════════════════════════════════════════════════

┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                          │
│   ❌ LOOKAHEAD BIAS (seeing future data during training)                                │
│      ✅ Prevented by: Purge/embargo in CV, chronological splits, train-only scaling    │
│                                                                                          │
│   ❌ DATA LEAKAGE (test info leaking into training)                                     │
│      ✅ Prevented by: Strict split boundaries, no shuffling, label_end_times purging   │
│                                                                                          │
│   ❌ SURVIVORSHIP BIAS (only analyzing winning trades)                                  │
│      ✅ Prevented by: All outcomes labeled (-1, 0, +1), no filtering by outcome        │
│                                                                                          │
│   ❌ OVERFITTING TO BACKTEST (optimizing for past performance)                          │
│      ✅ Prevented by: Walk-forward validation, PBO calculation, embargo periods        │
│                                                                                          │
│   ❌ REGIME BLINDNESS (assuming market is stationary)                                   │
│      ✅ Prevented by: Regime features, regime-aware evaluation, walk-forward windows   │
│                                                                                          │
└──────────────────────────────────────────────────────────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════════════════════════════
                                    QUICK REFERENCE
═══════════════════════════════════════════════════════════════════════════════════════════

# Run complete pipeline
./pipeline run --symbols MES

# Train individual model
python scripts/train_model.py --model xgboost --horizon 20
python scripts/train_model.py --model lstm --horizon 20 --seq-len 60

# Train ensemble (same-family models only!)
python scripts/train_model.py --model voting --base-models xgboost,lightgbm,catboost

# Run cross-validation
python scripts/run_cv.py --models xgboost --horizons 20 --n-splits 5

# Walk-forward validation
python scripts/run_walk_forward.py --model xgboost --horizon 20 --n-windows 10

# Check for overfitting
python scripts/run_cpcv_pbo.py --models xgboost,lightgbm --horizon 20
```
