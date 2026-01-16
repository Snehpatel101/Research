# Complete Codebase Analysis - 2026-01-15

## Executive Summary

**Serena reindex completed successfully.** The ML Model Factory is a production-grade, single-pipeline architecture for training, evaluating, and deploying ML models on OHLCV time series data.

---

## Project Structure

```
/home/jake/Desktop/Research/
├── src/
│   ├── phase1/           # Data Pipeline (Phases 1-5)
│   │   ├── stages/       # 17 pipeline stages
│   │   │   ├── ingest/
│   │   │   ├── clean/
│   │   │   ├── sessions/
│   │   │   ├── features/   # ~18 feature modules
│   │   │   ├── regime/
│   │   │   ├── mtf/        # Multi-timeframe
│   │   │   ├── labeling/   # Triple-barrier
│   │   │   ├── ga_optimize/ # Optuna
│   │   │   ├── final_labels/
│   │   │   ├── splits/
│   │   │   ├── scaling/
│   │   │   ├── datasets/   # Adapters (2D, 3D, 4D)
│   │   │   └── ...
│   │   ├── config/
│   │   └── utils/
│   ├── models/           # Model Factory (Phase 6)
│   │   ├── registry.py   # Plugin registry
│   │   ├── base.py       # BaseModel ABC
│   │   ├── trainer.py    # Re-exports training/
│   │   ├── training/     # Trainer implementation
│   │   ├── boosting/     # XGBoost, LightGBM, CatBoost
│   │   ├── neural/       # 10 neural models
│   │   ├── classical/    # RF, Logistic, SVM
│   │   ├── ensemble/     # Voting, Stacking, Blending + Meta-learners
│   │   └── config/       # Model configs
│   ├── cross_validation/ # CV & OOF Generation
│   │   ├── purged_kfold.py
│   │   ├── oof_generator.py
│   │   ├── cv_runner.py
│   │   └── ...
│   ├── feature_selection/
│   ├── common/           # Shared utilities
│   │   └── timeframes.py # Canonical timeframe definitions
│   ├── inference/        # Model serving
│   └── ...
├── config/
│   └── models/           # 24 YAML config files
├── scripts/              # CLI entry points
├── tests/                # Test suite
└── docs/                 # Documentation
```

---

## Model Registry (23 Models, 6 Families)

### Verification Status
- **ModelRegistry found at:** `src/models/registry.py`
- **Plugin decorator:** `@ModelRegistry.register(name, family, description, aliases)`
- **Total models:** 23 (22 if CatBoost unavailable)
- **Config files:** 24 YAML files in `config/models/`

### Models by Family

| Family | Models | Count | Status |
|--------|--------|-------|--------|
| **Boosting** | xgboost, lightgbm, catboost | 3 | ✅ Complete |
| **Classical** | random_forest, logistic, svm | 3 | ✅ Complete |
| **Neural** | lstm, gru, tcn, transformer, patchtst, itransformer, tft, nbeats, inceptiontime, resnet1d | 10 | ✅ Complete |
| **Ensemble** | voting, stacking, blending | 3 | ✅ Complete |
| **Meta-Learners** | ridge_meta, mlp_meta, calibrated_meta, xgboost_meta | 4 | ✅ Complete |

### Model Files
- `src/models/boosting/`: xgboost_model.py, lightgbm_model.py, catboost_model.py
- `src/models/classical/`: random_forest.py, logistic.py, svm.py
- `src/models/neural/`: lstm_model.py, gru_model.py, tcn_model.py, transformer_model.py, patchtst_model.py, itransformer_model.py, tft_model.py, nbeats_model.py, inceptiontime_model.py, resnet1d_model.py
- `src/models/ensemble/`: voting.py, stacking.py, blending.py, ridge_meta.py, mlp_meta.py, calibrated_meta.py, xgboost_meta.py

---

## Multi-Timeframe (MTF) Implementation

### Canonical Timeframes (src/common/timeframes.py)
```python
CANONICAL_TIMEFRAMES = [
    "1min", "5min", "10min", "15min", "20min", "25min", "30min", "45min", "60min"
]
```

### MTF Configuration (src/phase1/stages/mtf/constants.py)
- **DEFAULT_MTF_TIMEFRAMES:** 7 TFs (10m, 15m, 20m, 25m, 30m, 45m, 60m)
- **FULL_MTF_TIMEFRAMES:** 9 TFs (full canonical ladder including 1min, 5min)
- **MTF Modes:** BARS, INDICATORS, BOTH

### Status: ✅ Complete
- All 9 intraday timeframes implemented
- Upscaling from 1-min canonical source working
- MTF indicators generation functional

---

## Data Pipeline (Phases 1-5)

### Pipeline Stages (17 total in src/phase1/stages/)
1. **ingest/** - Load raw OHLCV
2. **clean/** - Resample, gap handling
3. **sessions/** - Session filtering
4. **features/** - 180+ indicators (momentum, trend, volatility, wavelets, microstructure)
5. **regime/** - Regime detection (volatility, trend, HMM)
6. **mtf/** - Multi-timeframe features
7. **labeling/** - Triple-barrier initial labels
8. **ga_optimize/** - Optuna parameter optimization
9. **final_labels/** - Apply optimized parameters
10. **splits/** - Train/val/test with purge/embargo
11. **scaling/** - Train-only robust scaling
12. **datasets/** - Build containers + adapters
13. **scaled_validation/** - Validate scaled data
14. **validation/** - Feature quality checks
15. **reporting/** - Generate reports
16. **meta_labeling/** - Meta-labeling (optional)

### Adapters (src/phase1/stages/datasets/adapters/)
- **Tabular adapter:** 2D (N, ~200 features)
- **Sequence adapter:** 3D (N, T, ~150 features)
- **Multi-Resolution adapter:** 4D (N, 9, T, 4) - `multi_resolution.py`

---

## Cross-Validation System

### Key Components (src/cross_validation/)
- **PurgedKFold:** Time-series CV with purge/embargo (`purged_kfold.py`)
- **OOFGenerator:** Out-of-fold predictions for stacking (`oof_generator.py`)
- **CVRunner:** Cross-validation orchestration (`cv_runner.py`)
- **CVTuner:** Optuna hyperparameter tuning (`cv_tuner.py`)

### OOF Generation
- `oof_core.py` - Core tabular OOF
- `oof_sequence.py` - Sequence model OOF
- `oof_stacking.py` - Stacking dataset builder

### CV Configuration
- Default: 70/15/15 train/val/test
- Purge: 60 bars (3× max horizon)
- Embargo: 1440 bars (~5 days at 5-min)

---

## Trainer Architecture

### Trainer (src/models/training/trainer.py)
- **Trainer class:** Main training orchestration
- **Methods:** `run()`, `_is_heterogeneous_ensemble()`, `_setup_tracker()`, etc.

### Heterogeneous Ensemble Support
- ✅ Implemented in trainer.py
- Dual data loading (2D for tabular + 3D for sequence models)
- Stacking with mixed model families

---

## Critical Bugs Fixed

All 5 critical bugs documented in `critical_bugs` memory have been fixed:
1. ✅ HMM Regime Detection - Lookahead bias (shift(1) applied)
2. ✅ GA Optimization - Test data leakage (safe_mode default)
3. ✅ Transaction Costs - Now in labels
4. ✅ MTF/Regime shift(1) - Applied at output
5. ✅ LightGBM num_leaves constraint - Validated

---

## Key Commands

```bash
# Data pipeline
./pipeline run --symbols MES
./pipeline run --symbols MES --process-all-timeframes  # 9 TFs

# Training
python scripts/train_model.py --model xgboost --horizon 20
python scripts/train_model.py --model lstm --horizon 20 --seq-len 30
python scripts/train_model.py --model stacking --horizon 20 \
  --base-models xgboost,lstm,patchtst --meta-learner ridge_meta

# Cross-validation
python scripts/run_cv.py --models xgboost --horizons 20 --n-splits 5

# List models
python scripts/train_model.py --list-models
```

---

## File Size Compliance
- Target: 650 lines
- Maximum: 800 lines (1300 acceptable if cohesive)

---

## Documentation Index
- Architecture: `docs/ARCHITECTURE.md`
- Phase guides: `docs/implementation/PHASE_*.md`
- Model reference: `docs/reference/MODELS.md`
- Quick reference: `docs/QUICK_REFERENCE.md`

---

**Last Updated:** 2026-01-15
