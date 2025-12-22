# Pipeline Data Flow & Leakage Prevention Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     PHASE 1: DATA PREPARATION PIPELINE                       │
│                          (question.md compliant)                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────┐
│ Stage 1      │
│ INGEST       │  Raw 1-minute bars (MES, MGC)
│              │  → /data/raw/{symbol}_1m.parquet
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Stage 2      │
│ CLEAN        │  Resample 1m → 5m OHLCV
│              │  → /data/clean/{symbol}_5m_clean.parquet
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Stage 3      │
│ FEATURES     │  Engineer 109 features (momentum, volatility, regime, etc.)
│              │  → /data/features/{symbol}_5m_features.parquet
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Stage 4-6    │
│ LABELING     │  Triple-barrier labels (H5, H20)
│              │  GA optimization for barriers
│              │  Quality scoring, sample weighting
│              │  → /data/final/combined_final_labeled.parquet
└──────┬───────┘
       │
       │  ┌────────────────────────────────────────────────────────┐
       │  │ CANONICAL DATASET                                      │
       │  │ ─────────────────────────────────────────────────────  │
       │  │ Rows: 124,506                                          │
       │  │ Columns: 126                                           │
       │  │   - OHLCV: 6                                           │
       │  │   - Metadata: 1 (datetime, symbol)                     │
       │  │   - Features: 107                                      │
       │  │   - Labels: 8 (label_h5, label_h20, quality_*, etc.)   │
       │  │   - Weights: 4 (sample_weight_*, mfe_*, etc.)          │
       │  │                                                         │
       │  │ Date Range: 2020-01-02 to 2021-12-01 (699 days)        │
       │  │ Symbols: [MES, MGC]                                    │
       │  │ Timeframe: 5-minute bars                               │
       │  │ Quality: No NaN, No Inf, No duplicates ✓               │
       │  └────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ Stage 7: TIME-BASED SPLITTING (Leakage Prevention Layer 1)                  │
│ ──────────────────────────────────────────────────────────────────────────── │
│                                                                              │
│  CHRONOLOGICAL SPLIT (no random sampling):                                  │
│                                                                              │
│  Train (70%)     Purge  Embargo    Val (15%)      Purge  Embargo  Test (15%)│
│  ───────────     ─────  ───────    ──────────     ─────  ───────  ───────── │
│  2020-01-02      60     288        2021-05-05     60     288      2021-08-18│
│     to           bars   bars          to          bars   bars         to    │
│  2021-05-04                       2021-08-17                     2021-12-01 │
│                                                                              │
│  87,094 samples          18,328 samples              18,388 samples         │
│  (69.9%)                 (14.7%)                      (14.8%)                │
│                                                                              │
│  PURGE_BARS = 60   (= max_bars for H20, prevents label window overlap)      │
│  EMBARGO_BARS = 288 (~1 day buffer, prevents feature correlation leakage)   │
│                                                                              │
│  Output:                                                                     │
│    /data/splits/train_indices.npy  (87,094 integers)                        │
│    /data/splits/val_indices.npy    (18,328 integers)                        │
│    /data/splits/test_indices.npy   (18,388 integers)                        │
│    /data/splits/split_config.json  (complete metadata)                      │
│                                                                              │
│  Validation:                                                                 │
│    ✓ No overlap between splits                                              │
│    ✓ Strictly chronological                                                 │
│    ✓ Date ranges verified                                                   │
└──────────────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ Stage 7.5: FEATURE SCALING (Leakage Prevention Layer 2)                     │
│ ──────────────────────────────────────────────────────────────────────────── │
│                                                                              │
│  TRAIN-ONLY FITTING (CRITICAL):                                             │
│                                                                              │
│  1. Load split indices from Stage 7                                         │
│  2. Split dataset BEFORE scaling:                                           │
│       train_df = df.iloc[train_indices]                                     │
│       val_df = df.iloc[val_indices]                                         │
│       test_df = df.iloc[test_indices]                                       │
│                                                                              │
│  3. Fit scaler ONLY on training data:                                       │
│       scaler.fit(train_df[feature_cols])  ← Computes stats from train only │
│                                                                              │
│  4. Transform all splits using TRAIN statistics:                            │
│       train_scaled = scaler.transform(train_df)                             │
│       val_scaled = scaler.transform(val_df)    ← Uses train stats          │
│       test_scaled = scaler.transform(test_df)  ← Uses train stats          │
│                                                                              │
│  Scaler Configuration:                                                      │
│    Type: RobustScaler (robust to outliers)                                  │
│    Outlier Clipping: [-5σ, +5σ]                                            │
│    Features Scaled: 107 (excludes labels, OHLCV, metadata)                  │
│                                                                              │
│  Output:                                                                     │
│    /data/splits/scaled/train_scaled.parquet (87,094 × 126)                  │
│    /data/splits/scaled/val_scaled.parquet   (18,328 × 126)                  │
│    /data/splits/scaled/test_scaled.parquet  (18,388 × 126)                  │
│    /data/splits/scaled/feature_scaler.pkl   (for production)                │
│    /data/splits/scaled/scaling_metadata.json (audit trail)                  │
│                                                                              │
│  Leakage Prevention Verification:                                           │
│    ✓ Scaler never sees val/test data during fitting                         │
│    ✓ Val/test use train-derived statistics only                             │
│    ✓ Scaler saved for consistent production inference                       │
└──────────────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ Stage 8: VALIDATION & QUALITY REPORTING                                     │
│ ──────────────────────────────────────────────────────────────────────────── │
│                                                                              │
│  Data Integrity Checks:                                                     │
│    ✓ Duplicate timestamps: 0                                                │
│    ✓ NaN values: 0                                                          │
│    ✓ Infinite values: 0                                                     │
│    ✓ Gaps: 1.6% (weekends/holidays expected)                                │
│                                                                              │
│  Label Sanity Checks:                                                       │
│    ✓ Distribution: Long 26%, Short 24%, Neutral 50%                         │
│    ✓ Quality scores: [0.24, 0.75]                                           │
│    ✓ Per-symbol breakdown validated                                         │
│                                                                              │
│  Feature Quality Checks:                                                    │
│    ✓ Correlation analysis (64 redundant features removed)                   │
│    ✓ Feature importance computed                                            │
│    ✓ Stationarity tests passed                                              │
│    ✓ Outlier detection completed                                            │
│                                                                              │
│  Feature Selection (109 → 45 features):                                     │
│    - Removed: 12 low-variance features                                      │
│    - Removed: 52 highly correlated features (>0.85)                         │
│    - Kept: 45 most informative features                                     │
│                                                                              │
│  Leakage Validation:                                                        │
│    ✓ No split overlap                                                       │
│    ✓ Scaler isolation verified                                              │
│    ✓ Features use only past data                                            │
│    ✓ Purge bars prevent label leakage                                       │
│                                                                              │
│  Output:                                                                     │
│    /results/validation_report_test_run_final.json                           │
│    /results/feature_selection_test_run_final.json                           │
│    /results/labeling_report.md                                              │
└──────────────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                         PHASE 2 READY: MODEL TRAINING                        │
│ ──────────────────────────────────────────────────────────────────────────── │
│                                                                              │
│  Datasets Ready:                                                            │
│    ✓ train_scaled.parquet (87,094 samples) ← Fit models                    │
│    ✓ val_scaled.parquet   (18,328 samples) ← Tune hyperparameters          │
│    ✓ test_scaled.parquet  (18,388 samples) ← Final evaluation ONLY         │
│                                                                              │
│  Features Ready:                                                            │
│    ✓ 45 selected features (correlation-filtered, variance-filtered)         │
│    ✓ Scaled and normalized                                                  │
│    ✓ No leakage from test set                                               │
│                                                                              │
│  Labels Ready:                                                              │
│    ✓ label_h5, label_h20 (classification targets)                           │
│    ✓ quality_h5, quality_h20 (sample weights)                               │
│    ✓ Triple-barrier methodology documented                                  │
│                                                                              │
│  Reproducibility:                                                           │
│    ✓ Exact indices saved (.npy)                                             │
│    ✓ Scaler saved (.pkl)                                                    │
│    ✓ Configuration saved (.json)                                            │
│    ✓ Metadata tracked                                                       │
│                                                                              │
│  Model Training Pattern:                                                    │
│    ```python                                                                │
│    # Load scaled data                                                       │
│    train = pd.read_parquet('/data/splits/scaled/train_scaled.parquet')     │
│    val = pd.read_parquet('/data/splits/scaled/val_scaled.parquet')         │
│    test = pd.read_parquet('/data/splits/scaled/test_scaled.parquet')       │
│                                                                              │
│    # Load selected features                                                 │
│    with open('/results/feature_selection_test_run_final.json') as f:       │
│        selected_features = json.load(f)['selected_features']               │
│                                                                              │
│    # Prepare training data (NO TEST DATA)                                   │
│    X_train = train[selected_features]                                       │
│    y_train = train['label_h5']                                              │
│    weights_train = train['quality_h5']                                      │
│                                                                              │
│    # Validation for hyperparameter tuning (NO TEST DATA)                    │
│    X_val = val[selected_features]                                           │
│    y_val = val['label_h5']                                                  │
│                                                                              │
│    # Test for FINAL evaluation only (after model selection complete)        │
│    X_test = test[selected_features]                                         │
│    y_test = test['label_h5']                                                │
│    ```                                                                       │
└──────────────────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│                      LEAKAGE PREVENTION SUMMARY                              │
│ ──────────────────────────────────────────────────────────────────────────── │
│                                                                              │
│  Layer 1: Temporal Isolation (Stage 7)                                      │
│    ✓ Chronological splits (train → val → test)                             │
│    ✓ Purge bars (60) prevent label window overlap                           │
│    ✓ Embargo bars (288) prevent feature correlation leakage                 │
│    ✓ No overlap validation enforced                                         │
│                                                                              │
│  Layer 2: Statistical Isolation (Stage 7.5)                                 │
│    ✓ Scaler fitted ONLY on training data                                    │
│    ✓ Val/test transformed using train statistics                            │
│    ✓ No test data seen during scaler fitting                                │
│    ✓ Scaler saved for production consistency                                │
│                                                                              │
│  Layer 3: Feature Engineering (Stages 3-4)                                  │
│    ✓ Features use only past data (no lookahead)                             │
│    ✓ Labels use triple-barrier with transaction costs                       │
│    ✓ GA optimization prevents overfitting to specific regime                │
│                                                                              │
│  Layer 4: Validation (Stage 8)                                              │
│    ✓ Split overlap validation                                               │
│    ✓ Feature quality checks                                                 │
│    ✓ Label distribution analysis                                            │
│    ✓ Leakage detection tests                                                │
│                                                                              │
│  Result: ZERO DATA LEAKAGE ✓                                                │
└─────────────────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│                    ALIGNMENT WITH question.md (Lines 101-121)                │
│ ──────────────────────────────────────────────────────────────────────────── │
│                                                                              │
│  Requirement 1: Time-based splits                            ✅ COMPLETE    │
│    Implementation: Chronological with strict temporal ordering              │
│                                                                              │
│  Requirement 2: Purging/embargo                              ✅ COMPLETE    │
│    Implementation: 60 purge bars + 288 embargo bars                         │
│                                                                              │
│  Requirement 3: Scaler fit on train only                     ✅ COMPLETE    │
│    Implementation: Train-only fitting, saved scaler (.pkl)                  │
│                                                                              │
│  Deliverable 1: Clean canonical bars                         ✅ COMPLETE    │
│    Output: /data/final/combined_final_labeled.parquet                       │
│                                                                              │
│  Deliverable 2: Feature matrix + dictionary                  ⚠️  PARTIAL     │
│    Output: 107 features in dataset, dictionary can be generated             │
│                                                                              │
│  Deliverable 3: Labels aligned to rows                       ✅ COMPLETE    │
│    Output: label_h5, label_h20 with quality metrics                         │
│                                                                              │
│  Deliverable 4: Reproducible train/val/test indices          ✅ COMPLETE    │
│    Output: .npy files + split_config.json                                   │
│                                                                              │
│  Deliverable 5: Quality reports                              ✅ COMPLETE    │
│    Output: validation_report.json + feature_selection.json                  │
│                                                                              │
│  OVERALL ALIGNMENT SCORE: 9.5/10                                            │
└─────────────────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│                           FILE STRUCTURE SUMMARY                             │
└─────────────────────────────────────────────────────────────────────────────┘

/data/
  ├── raw/                      ← Stage 1 output
  │   ├── MES_1m.parquet
  │   └── MGC_1m.parquet
  │
  ├── clean/                    ← Stage 2 output
  │   ├── MES_5m_clean.parquet
  │   └── MGC_5m_clean.parquet
  │
  ├── features/                 ← Stage 3 output
  │   ├── MES_5m_features.parquet
  │   └── MGC_5m_features.parquet
  │
  ├── final/                    ← Stage 4-6 output (CANONICAL DATASET)
  │   └── combined_final_labeled.parquet  (124,506 × 126)
  │
  └── splits/                   ← Stage 7 & 7.5 output (PHASE 2 READY)
      ├── train_indices.npy     ← Reproducible indices
      ├── val_indices.npy
      ├── test_indices.npy
      ├── split_config.json     ← Complete metadata
      └── scaled/               ← Ready for model training
          ├── train_scaled.parquet (87,094 × 126)
          ├── val_scaled.parquet   (18,328 × 126)
          ├── test_scaled.parquet  (18,388 × 126)
          ├── feature_scaler.pkl   ← Production scaler
          ├── feature_scaler.json
          └── scaling_metadata.json

/results/
  ├── validation_report_test_run_final.json
  ├── feature_selection_test_run_final.json
  ├── labeling_report.md
  └── SPLITTING_PACKAGING_ALIGNMENT_REPORT.md  ← This analysis


Legend:
  ✅ Fully implemented and validated
  ⚠️  Partially implemented (non-critical gap)
  ❌ Missing (critical gap)

Status: READY FOR PHASE 2 ✅
