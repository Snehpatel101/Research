# ML Factory Overview - TOPSTEPX Single-Contract Trading

## North Star

**Production-grade, model-agnostic ML factory for systematic futures trading research and deployment.**

- **Input**: Single canonical OHLCV dataset for one futures contract (MES, MGC)
- **Output**: Reproducible training datasets, trained model artifacts, evaluation reports, inference-ready predictors

## What "Factory" Means

1. **ONE canonical pipeline**: Single data flow from raw OHLCV to trained models (NOT separate pipelines)
2. **Deterministic adapters**: Model-family specific shape transformations (2D/3D/4D) without feature engineering
3. **Single source of truth**: Canonical dataset stored once, all models consume identical features/labels
4. **Fair comparison**: Same splits, same leakage prevention, same metrics/backtest assumptions
5. **Research → inference parity**: Exact same preprocessing graph at training and runtime

## Architecture (8 Phases, ONE Pipeline)

### Phases 1-5: Data Pipeline (`src/phase1/`)
```
Phase 1: Raw OHLCV → Ingest → Clean → Validate
Phase 2: MTF Upscaling → 9 timeframes (⚠️ 5 of 9 implemented)
Phase 3: Features → ~180 features (base indicators + wavelets + microstructure + MTF)
Phase 4: Labels → Triple-barrier + Splits (70/15/15) + Scaling
Phase 5: Adapters → 2D (tabular), 3D (sequence), 4D (multi-res, planned)
```

**Output**: Canonical dataset in `data/splits/scaled/` (single source of truth)

**Leakage protection**: Purge (60), embargo (1440), shift(1) on MTF, train-only scaling

### Phase 6: Model Training (`src/models/`)
```
13 models (implemented): Boosting (3), Neural (4), Classical (3), Ensemble (3)
6 models (planned): CNN (2), Advanced Transformers (3), MLP (1)
```

**Model Interface**: `BaseModel` with `fit()`, `predict()`, `save()`, `load()`
**Registry**: Plugin-based `@register()` decorator system

### Phase 7: Cross-Validation (`src/cross_validation/`)
```
PurgedKFold → OOF Generation → Feature Selection → Hyperparameter Tuning → Stacking
```

**Key Features**: Time-series aware CV, purge/embargo, walk-forward validation

### Phase 8: Meta-Learners (Planned)
```
Ensemble predictions → Regime-aware/Confidence-based/Adaptive weighting → Final predictions
```

**Purpose**: Dynamic ensemble selection based on market context

## Single-Contract Isolation (NON-NEGOTIABLE)

- **One contract per pipeline run** (MES or MGC, never both)
- **Complete data isolation** - no cross-symbol correlation or features
- **Symbol configurability** via config: `./pipeline run --symbols MES`

## Model Families & Compatibility

| Family | Models | Input Shape | Ensemble Compatible With |
|--------|--------|-------------|--------------------------|
| Boosting | xgboost, lightgbm, catboost | 2D (n_samples, n_features) | Other tabular only |
| Classical | random_forest, logistic, svm | 2D (n_samples, n_features) | Other tabular only |
| Neural | lstm, gru, tcn, transformer | 3D (n_samples, seq_len, n_features) | Other sequence only |
| Ensemble | voting, stacking, blending | Varies | Same-family base models |

**CRITICAL**: Mixed tabular+sequence ensembles are NOT supported and will raise `EnsembleCompatibilityError`.

### Pipeline Implementation Status

| Phase | Component | Status | Gap |
|-------|-----------|--------|-----|
| 1 | OHLCV Ingestion | ✅ Complete | None |
| 2 | MTF Upscaling | ⚠️ Partial | 4 of 9 timeframes missing (5min, 10min, 20min, 25min, 45min) |
| 3 | Feature Engineering | ✅ Complete | MTF features only from 5 timeframes (intended: 9) |
| 4 | Labeling + Splits | ✅ Complete | None |
| 5 | Adapters | ⚠️ Partial | Multi-res adapter (4D) not implemented |
| 6 | Model Training | ⚠️ Partial | 6 advanced models not implemented (CNN, transformers, MLP) |
| 7 | Cross-Validation | ✅ Complete | None |
| 8 | Meta-Learners | ❌ Not Started | All meta-learner strategies not implemented |

**Priority Tasks**:
1. Complete 9-timeframe MTF ladder (1-2 days)
2. Implement multi-resolution adapter (3 days)
3. Add advanced models (14-18 days)
4. Build meta-learners (5-7 days)

## Key Parameters

```python
SYMBOL = 'MES'  # One symbol per run
LABEL_HORIZONS = [5, 10, 15, 20]  # Prediction horizons in bars
TRAIN/VAL/TEST = 70/15/15
PURGE_BARS = 60  # 3× max_horizon (prevents label leakage)
EMBARGO_BARS = 1440  # ~5 days at 5-min (prevents serial correlation)
```

## File Size Limits

- **Target**: 650 lines per file
- **Maximum**: 800 lines per file
- Beyond 800 → refactor and split responsibilities
