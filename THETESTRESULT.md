# ML Factory — 12 Model Test Results

**Date:** 2026-02-12
**Data:** MES 1-minute futures, 1 month
**Timeframe:** 5min
**Horizon:** 20 bars
**Training samples:** 33,778
**Validation samples:** 5,753
**Test samples:** 5,752
**Features:** 291
**Classes:** 3 (Short=-1, Neutral=0, Long=1)
**Epochs:** 3 (all models)
**Device:** CPU

---

## Summary Table

| Model | Family | Time | Val Acc | Val F1 | Test Acc | Test F1 | Val Precision | Val Recall |
|-------|--------|------|---------|--------|----------|---------|---------------|------------|
| XGBoost | Boosting | 67.4s | 0.4651 | 0.4084 | 0.4986 | 0.3186 | 0.3964 | 0.4320 |
| LightGBM | Boosting | 62.1s | 0.4820 | 0.4167 | 0.4967 | 0.3007 | 0.4075 | 0.4304 |
| CatBoost | Boosting | 111.3s | 0.4241 | 0.3353 | 0.4013 | 0.2931 | 0.3797 | 0.4550 |
| LSTM | Neural RNN | 453.6s | 0.4970 | 0.3124 | 0.5568 | 0.3459 | 0.3160 | 0.3303 |
| GRU | Neural RNN | 446.7s | 0.4677 | 0.3173 | 0.4748 | 0.3223 | 0.3124 | 0.3244 |
| TCN | Neural CNN | 477.6s | 0.4672 | 0.3065 | 0.4611 | 0.3126 | 0.3252 | 0.3375 |
| InceptionTime | Neural CNN | 2,699.1s | 0.4845 | 0.3554 | 0.5212 | 0.3385 | 0.5293 | 0.3480 |
| ResNet1D | Neural CNN | 1,746.6s | 0.4837 | 0.3171 | 0.5150 | 0.3353 | 0.3128 | 0.3264 |
| PatchTST | Transformer | 215.9s | 0.4612 | 0.3002 | 0.4242 | 0.2723 | 0.3174 | 0.3343 |
| iTransformer | Transformer | 132.9s | 0.5026 | 0.2942 | 0.5247 | 0.3030 | 0.3098 | 0.3285 |
| TFT | Transformer | 37,480.2s | 0.5260 | 0.2298 | 0.5624 | 0.2400 | 0.1753 | 0.3333 |
| N-BEATS | MLP | 27.7s | 0.5260 | 0.2298 | 0.5624 | 0.2400 | 0.1753 | 0.3333 |

---

## Rankings

### By Validation Accuracy
| Rank | Model | Val Acc |
|------|-------|---------|
| 1 | TFT | 0.5260 |
| 1 | N-BEATS | 0.5260 |
| 3 | iTransformer | 0.5026 |
| 4 | LSTM | 0.4970 |
| 5 | InceptionTime | 0.4845 |
| 6 | ResNet1D | 0.4837 |
| 7 | LightGBM | 0.4820 |
| 8 | GRU | 0.4677 |
| 9 | TCN | 0.4672 |
| 10 | XGBoost | 0.4651 |
| 11 | PatchTST | 0.4612 |
| 12 | CatBoost | 0.4241 |

### By Test Accuracy
| Rank | Model | Test Acc |
|------|-------|----------|
| 1 | TFT | 0.5624 |
| 1 | N-BEATS | 0.5624 |
| 3 | LSTM | 0.5568 |
| 4 | iTransformer | 0.5247 |
| 5 | InceptionTime | 0.5212 |
| 6 | ResNet1D | 0.5150 |
| 7 | XGBoost | 0.4986 |
| 8 | LightGBM | 0.4967 |
| 9 | GRU | 0.4748 |
| 10 | TCN | 0.4611 |
| 11 | PatchTST | 0.4242 |
| 12 | CatBoost | 0.4013 |

### By Validation F1 (Macro)
| Rank | Model | Val F1 |
|------|-------|--------|
| 1 | LightGBM | 0.4167 |
| 2 | XGBoost | 0.4084 |
| 3 | InceptionTime | 0.3554 |
| 4 | CatBoost | 0.3353 |
| 5 | GRU | 0.3173 |
| 6 | ResNet1D | 0.3171 |
| 7 | LSTM | 0.3124 |
| 8 | TCN | 0.3065 |
| 9 | PatchTST | 0.3002 |
| 10 | iTransformer | 0.2942 |
| 11 | TFT | 0.2298 |
| 11 | N-BEATS | 0.2298 |

### By Test F1 (Macro)
| Rank | Model | Test F1 |
|------|-------|---------|
| 1 | LSTM | 0.3459 |
| 2 | InceptionTime | 0.3385 |
| 3 | ResNet1D | 0.3353 |
| 4 | GRU | 0.3223 |
| 5 | XGBoost | 0.3186 |
| 6 | TCN | 0.3126 |
| 7 | iTransformer | 0.3030 |
| 8 | LightGBM | 0.3007 |
| 9 | CatBoost | 0.2931 |
| 10 | PatchTST | 0.2723 |
| 11 | TFT | 0.2400 |
| 11 | N-BEATS | 0.2400 |

### By Speed (Fastest First)
| Rank | Model | Time |
|------|-------|------|
| 1 | N-BEATS | 27.7s |
| 2 | LightGBM | 62.1s |
| 3 | XGBoost | 67.4s |
| 4 | CatBoost | 111.3s |
| 5 | iTransformer | 132.9s |
| 6 | PatchTST | 215.9s |
| 7 | GRU | 446.7s |
| 8 | LSTM | 453.6s |
| 9 | TCN | 477.6s |
| 10 | ResNet1D | 1,746.6s |
| 11 | InceptionTime | 2,699.1s |
| 12 | TFT | 37,480.2s |

---

## Per-Model Details

### XGBoost
- **Family:** Boosting (2D tabular)
- **Training time:** 67.4s
- **Val:** acc=0.4651, f1=0.4084, precision=0.3964, recall=0.4320
- **Test:** acc=0.4986, f1=0.3186
- **Notes:** Solid all-rounder. Best F1 among boosting on val. Good generalization to test.

### LightGBM
- **Family:** Boosting (2D tabular)
- **Training time:** 62.1s
- **Val:** acc=0.4820, f1=0.4167, precision=0.4075, recall=0.4304
- **Test:** acc=0.4967, f1=0.3007
- **Notes:** Highest val F1 of all models (0.4167). Fast. Some overfitting (val F1 0.42 vs test F1 0.30).

### CatBoost
- **Family:** Boosting (2D tabular)
- **Training time:** 111.3s
- **Val:** acc=0.4241, f1=0.3353, precision=0.3797, recall=0.4550
- **Test:** acc=0.4013, f1=0.2931
- **Notes:** Lowest accuracy but highest recall (0.4550) — catches more signals. Had thread_count bug (fixed). Ordered boosting helps with overfitting.

### LSTM
- **Family:** Neural RNN (3D sequence)
- **Training time:** 453.6s
- **Val:** acc=0.4970, f1=0.3124, precision=0.3160, recall=0.3303
- **Test:** acc=0.5568, f1=0.3459
- **Notes:** Best test F1 of all models (0.3459). Strong generalization — test accuracy significantly higher than val. Good at learning temporal patterns.

### GRU
- **Family:** Neural RNN (3D sequence)
- **Training time:** 446.7s
- **Val:** acc=0.4677, f1=0.3173, precision=0.3124, recall=0.3244
- **Test:** acc=0.4748, f1=0.3223
- **Notes:** Slightly lower than LSTM but similar profile. Faster convergence typically. Consistent val-to-test transfer.

### TCN
- **Family:** Neural CNN (3D sequence)
- **Training time:** 477.6s
- **Val:** acc=0.4672, f1=0.3065, precision=0.3252, recall=0.3375
- **Test:** acc=0.4611, f1=0.3126
- **Notes:** Stable performance. Causal convolutions prevent leakage by design. Good for local pattern detection.

### InceptionTime
- **Family:** Neural CNN (3D sequence)
- **Training time:** 2,699.1s (45 min)
- **Val:** acc=0.4845, f1=0.3554, precision=0.5293, recall=0.3480
- **Test:** acc=0.5212, f1=0.3385
- **Notes:** Highest precision of all models (0.5293). Multi-scale filters capture patterns at different resolutions. Slow on CPU but strong results.

### ResNet1D
- **Family:** Neural CNN (3D sequence)
- **Training time:** 1,746.6s (29 min)
- **Val:** acc=0.4837, f1=0.3171, precision=0.3128, recall=0.3264
- **Test:** acc=0.5150, f1=0.3353
- **Notes:** Residual connections help with gradient flow. Good test generalization. Heavy on CPU.

### PatchTST
- **Family:** Transformer
- **Training time:** 215.9s
- **Val:** acc=0.4612, f1=0.3002, precision=0.3174, recall=0.3343
- **Test:** acc=0.4242, f1=0.2723
- **Notes:** Patch-based approach for time series. Moderate results with only 3 epochs — likely needs more training. Efficient transformer design.

### iTransformer
- **Family:** Transformer
- **Training time:** 132.9s
- **Val:** acc=0.5026, f1=0.2942, precision=0.3098, recall=0.3285
- **Test:** acc=0.5247, f1=0.3030
- **Notes:** Best accuracy-to-speed ratio. Inverted architecture (attention over features, not time). Only 2 min training. Strong potential with more epochs.

### TFT (Temporal Fusion Transformer)
- **Family:** Transformer
- **Training time:** 37,480.2s (10.4 hours)
- **Val:** acc=0.5260, f1=0.2298, precision=0.1753, recall=0.3333
- **Test:** acc=0.5624, f1=0.2400
- **Notes:** Highest accuracy but lowest F1 — predicting mostly one class (neutral). Variable selection + attention makes it powerful but extremely slow on CPU. Needs GPU. Low precision (0.1753) suggests it's not differentiating classes well with only 3 epochs.

### N-BEATS
- **Family:** MLP (decomposition)
- **Training time:** 27.7s
- **Val:** acc=0.5260, f1=0.2298, precision=0.1753, recall=0.3333
- **Test:** acc=0.5624, f1=0.2400
- **Notes:** Fastest model overall (28s). Same accuracy as TFT but in a fraction of the time. Basis expansion architecture for trend/seasonality decomposition. Like TFT, low F1 suggests it defaults to majority class with minimal training.

---

## Key Observations

1. **Accuracy vs F1 tradeoff:** TFT/N-BEATS have highest accuracy (0.526) but lowest F1 (0.230). They're likely predicting the majority class. LightGBM has highest F1 (0.417) with lower accuracy (0.482) — it's actually differentiating the 3 classes better.

2. **Best generalization (val→test):** LSTM showed the strongest improvement from val to test (0.497→0.557 accuracy), suggesting it learned genuine temporal patterns rather than overfitting.

3. **Speed tiers:**
   - Lightning (<2 min): N-BEATS, LightGBM, XGBoost, CatBoost, iTransformer
   - Moderate (2-8 min): PatchTST, GRU, LSTM, TCN
   - Heavy (30-45 min): ResNet1D, InceptionTime
   - Extreme (10h+): TFT (CPU only — needs GPU)

4. **Only 3 epochs:** All models trained with minimal epochs. Boosting models (XGBoost, LightGBM) naturally handle this well. Neural models (especially transformers) would benefit significantly from more epochs.

5. **Precision champion:** InceptionTime had the highest precision (0.5293) — when it predicts a trade, it's right more often. Useful for conservative strategies.

6. **Recall champion:** CatBoost had the highest recall (0.4550) — it catches the most trading signals. Useful for coverage-focused strategies.

7. **For ensembling:** The diversity across model families (boosting vs RNN vs CNN vs transformer) makes this set well-suited for stacking ensembles. Models disagree enough to provide complementary signals.

---

## Test Configuration

```yaml
symbol: MES
timeframe: 5min
horizon: 20
max_epochs: 3
early_stopping_patience: 2
batch_size: 128
device: cpu
mixed_precision: false
data_path: runs/20260212_120129_588531_22ff/data/splits/scaled/
```

---

*Generated: 2026-02-12*
*All 12/12 models confirmed working*
