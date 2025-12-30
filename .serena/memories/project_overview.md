# ML Factory Overview - TOPSTEPX Single-Contract Trading

## North Star

**Production-grade, model-agnostic ML factory for systematic futures trading research and deployment.**

- **Input**: Single canonical OHLCV dataset for one futures contract (MES, MGC)
- **Output**: Reproducible training datasets, trained model artifacts, evaluation reports, inference-ready predictors

## What "Factory" Means

1. **One data source → many derived datasets**: Deterministic resampling and rational MTF feature generation
2. **Model-family specific adapters**: Tabular vs sequence models require different representations (NOT one-size-fits-all)
3. **Fair comparison**: Same splits, same leakage prevention, same metrics/backtest assumptions
4. **Research → inference parity**: Exact same preprocessing graph at training and runtime

## Architecture (3 Phases)

### Phase 1: Data Pipeline (`src/phase1/`)
```
Raw OHLCV → Clean/Resample → Features (150+) → Regime Detection → Labels → Splits → Scaling → Datasets
```

**Key Stages**: ingest → clean → sessions → features → regime → mtf → labeling → ga_optimize → final_labels → splits → scaling → datasets → validation

### Phase 2: Model Factory (`src/models/`)
```
13 models across 4 families: Boosting (3), Neural (4), Classical (3), Ensemble (3)
```

**Model Interface**: `BaseModel` with `fit()`, `predict()`, `save()`, `load()`
**Registry**: Plugin-based `@register()` decorator system

### Phase 3: Cross-Validation (`src/cross_validation/`)
```
PurgedKFold → OOF Generation → Feature Selection → Hyperparameter Tuning → Stacking
```

**Key Features**: Time-series aware CV, purge/embargo, walk-forward validation

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
