# Phase C: Preprocessing Migration (Stages 10-12)

**Status:** Planning Complete
**Estimated Effort:** 4 days

---

## Current State Summary

### Stage 10: Splits
| Location | Lines | Purpose |
|----------|-------|---------|
| `src/phase1/stages/splits/core.py` | 321 | Chronological splits with purge/embargo |
| `src/phase1/stages/splits/run.py` | 208 | Pipeline wrapper |

### Stage 11: Scaling
| Location | Lines | Purpose |
|----------|-------|---------|
| `src/phase1/stages/scaling/scaler.py` | 535 | FeatureScaler (train-only) |
| `src/phase1/stages/scaling/core.py` | 327 | ScalerConfig, types |
| `src/phase1/stages/scaling/run.py` | 362 | Pipeline wrapper |

### Stage 12: Adapters
| Location | Lines | Purpose |
|----------|-------|---------|
| `src/adapters/tabular.py` | 255 | 2D adapter (N, F) |
| `src/adapters/sequence.py` | 435 | 3D adapter (N, T, F) |
| `src/phase1/stages/datasets/adapters/multi_resolution.py` | 619 | 4D adapter (N, TF, T, 4) |
| `src/adapters/registry.py` | 190 | Contract-based routing |

---

## Target State

### In `src/pipeline/phases/training.py` (Stages 10-12 portion)

```python
class Stage10Splits:
    """70/15/15 splits with purge_bars=60, embargo_bars=1440."""

    def run(self, df: pd.DataFrame) -> SplitsResult:
        from src.phase1.stages.splits.core import create_chronological_splits
        return create_chronological_splits(df, **self.config)

class Stage11Scaling:
    """Train-only RobustScaler fitting."""

    def run(self, splits: SplitsResult) -> ScalingResult:
        from src.phase1.stages.scaling.scaler import FeatureScaler
        scaler = FeatureScaler().fit(splits.train_df)
        return ScalingResult(
            train_scaled=scaler.transform(splits.train_df),
            val_scaled=scaler.transform(splits.val_df),
            test_scaled=scaler.transform(splits.test_df),
            scaler=scaler,
        )

class Stage12Adaptation:
    """Model-family adapters: Tabular (2D), Sequence (3D), MultiRes (4D)."""

    def run(self, scaling: ScalingResult, model_family: str) -> AdaptationResult:
        adapter = AdapterRegistry.get_for_model(model_family)
        return adapter.transform(scaling)
```

---

## Adapter Contracts

| Adapter | Output Shape | Models |
|---------|--------------|--------|
| TabularAdapter | (N, F) 2D | XGBoost, LightGBM, CatBoost, RF, Logistic, SVM |
| SequenceAdapter | (N, T, F) 3D | LSTM, GRU, TCN, Transformer |
| MultiResolutionAdapter | (N, 9, T, 4) 4D | PatchTST, iTransformer, TFT, N-BEATS, InceptionTime, ResNet1D |

### Model Family Mapping
```python
MODEL_ADAPTER_MAP = {
    "xgboost": "tabular", "lightgbm": "tabular", "catboost": "tabular",
    "random_forest": "tabular", "logistic": "tabular", "svm": "tabular",
    "lstm": "sequence", "gru": "sequence", "tcn": "sequence", "transformer": "sequence",
    "patchtst": "multi_stream", "itransformer": "multi_stream", "tft": "multi_stream",
    "nbeats": "multi_stream", "inceptiontime": "multi_stream", "resnet1d": "multi_stream",
}
```

---

## Leakage Prevention

### Stage 10: Purge and Embargo
```
|---------- TRAIN ----------|--P--|---E---|------ VAL ------|--P--|---E---|---- TEST ---|
                            purge embargo                    purge embargo
```
- **Purge (60 bars):** Removes label overlap at boundaries
- **Embargo (1440 bars):** ~5 trading days buffer

### Stage 11: Train-Only Scaling
- `scaler.fit()` called ONLY on train_df
- `scaler.transform()` applied to all splits with SAME parameters
- Scaler statistics (median, IQR) from train only

### Stage 12: Sequence Leakage Prevention
- Labels at window END (time t), not future
- Sequences never span symbol boundaries

---

## Interface Contracts

| Stage | Input | Output |
|-------|-------|--------|
| 10 | Optimized df from Stage 9 | `SplitsResult(train, val, test indices)` |
| 11 | SplitsResult | `ScalingResult(scaled dfs, scaler.pkl)` |
| 12 | ScalingResult | `AdaptationResult(2D/3D/4D tensors per model)` |

**Checkpoint:** `data/splits/scaled/{symbol}_{split}.parquet`

---

## Migration Steps

1. **Define interfaces** (0.5 days): SplitsConfig, ScalingConfig, AdaptationConfig
2. **Wrap Stage 10** (0.5 days): Import create_chronological_splits()
3. **Wrap Stage 11** (0.5 days): Import FeatureScaler
4. **Consolidate Stage 12** (1 day): Unify adapters from two locations
5. **Integration** (0.5 days): Chain stages, add checkpoints
6. **Testing** (1 day): Unit + leakage validation tests

---

## Critical Files

1. `src/phase1/stages/splits/core.py` - Splits with purge/embargo
2. `src/phase1/stages/scaling/scaler.py` - FeatureScaler class
3. `src/adapters/base.py` - BaseAdapter and AdapterResult
4. `src/adapters/registry.py` - Model-to-adapter routing
5. `src/phase1/stages/datasets/adapters/multi_resolution.py` - 4D adapter (619 lines)
