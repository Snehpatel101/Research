# ML Factory - Remaining Tasks

**Last Updated:** 2026-01-24
**Status:** Phases 0-5 Complete | Advanced Models Pending

---

## Completed (see COMPLETION.md)

| Phase | Impact |
|-------|--------|
| Phase 0 | -5,336 lines (deduplication) |
| Phase 1 | +616 lines (contract enforcement) |
| Phase 2 | +958 lines (4D infrastructure) |
| Phase 3 | +2,298 lines (5D Optuna) |
| Phase 4 | +50 lines (validation wiring) |
| Phase 5 | +1,281 lines (MLFactory + ExperimentConfig) |

---

## Remaining: Advanced Models

| Model | Status |
|-------|--------|
| InceptionTime | ⬜ |
| 1D ResNet | ⬜ |
| PatchTST | ⬜ |
| iTransformer | ⬜ |
| TFT | ⬜ |
| N-BEATS | ⬜ |

---

### 3D Models (Sequence Adapter)

#### InceptionTime ⬜
**Location:** `src/models/neural/inception_time.py`
**Contract:** `DataRank.SEQUENCE_3D`, `FeatureMode.ENGINEERED`

```python
# Implementation pattern
class InceptionTimeModel(BaseNeuralModel):
    """InceptionTime for time series classification."""
    # Multiple inception modules with different kernel sizes
    # Residual connections
    # Global average pooling
```

#### 1D ResNet ⬜
**Location:** `src/models/neural/resnet_1d.py`
**Contract:** `DataRank.SEQUENCE_3D`, `FeatureMode.ENGINEERED`

```python
# Implementation pattern
class ResNet1DModel(BaseNeuralModel):
    """1D ResNet for time series classification."""
    # Residual blocks with 1D convolutions
    # Batch normalization
    # Skip connections
```

---

### 4D Models (MultiStream Adapter)

#### PatchTST ⬜
**Location:** `src/models/neural/patchtst.py`
**Contract:** `DataRank.MULTI_TF_4D`, `FeatureMode.RAW`

```python
# Implementation pattern
class PatchTSTModel(BaseNeuralModel):
    """Patch Time Series Transformer."""
    # Patching mechanism
    # Channel-independent processing
    # Transformer encoder
```

#### iTransformer ⬜
**Location:** `src/models/neural/itransformer.py`
**Contract:** `DataRank.MULTI_TF_4D`, `FeatureMode.RAW`

```python
# Implementation pattern
class iTransformerModel(BaseNeuralModel):
    """Inverted Transformer - attention over features."""
    # Inverted attention (variate tokens)
    # Feed-forward on time dimension
```

#### TFT (Temporal Fusion Transformer) ⬜
**Location:** `src/models/neural/tft.py`
**Contract:** `DataRank.MULTI_TF_4D`, `FeatureMode.RAW`

```python
# Implementation pattern
class TFTModel(BaseNeuralModel):
    """Temporal Fusion Transformer."""
    # Variable selection networks
    # LSTM encoder
    # Multi-head attention
    # Gated residual connections
```

#### N-BEATS ⬜
**Location:** `src/models/neural/nbeats.py`
**Contract:** `DataRank.MULTI_TF_4D`, `FeatureMode.RAW`

```python
# Implementation pattern
class NBEATSModel(BaseNeuralModel):
    """Neural Basis Expansion Analysis."""
    # Stacks of blocks
    # Trend and seasonality decomposition
    # Fully connected layers
```

---

## Deferred (Low Priority)

| Task | Description |
|------|-------------|
| 5C | Unified deployment bundle (tar.gz format) |
| 4C | Ensemble diversity analysis integration |
| 4D | Deflated Sharpe Ratio post-Optuna validation |
| 4E | Bootstrap CIs in financial reports |
| 4F | Auto calibration in orchestrator |
| 4G | Bet sizing connection to backtest |
| - | MTF ablation flag |

---

## After Each Model - REQUIRED

### 1. Verify
```bash
python -c "from src.models.neural.<model> import <ModelClass>; print('OK')"
python -c "from src.core.contracts import get_model_contract; print(get_model_contract('<model>'))"
ruff check src/models/neural/<model>.py
black src/models/neural/<model>.py
```

### 2. Update Docs
1. Change ⬜ to ✅ in table above
2. Change ⬜ to ✅ in model header
3. Change ⬜ to ✅ in CLEANUP_PLAN.md table
4. After all 6 done → add "Advanced Models" to COMPLETION.md

---

*For completed phase details, see COMPLETION.md*
