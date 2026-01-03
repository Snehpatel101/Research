# Feature Selection Research for Heterogeneous ML Factory

**Date:** 2026-01-03
**Research by:** 4 Specialized ML Agents

---

## Executive Summary

**Key Finding: NO, all models should NOT get 180 features.**

Research strongly supports **per-model feature selection** where each model family receives features tailored to its inductive biases. This maximizes prediction diversity and ensemble performance.

### Recommended Feature Allocation

| Model Family | Primary TF | Feature Count | Feature Type |
|--------------|-----------|---------------|--------------|
| **CatBoost/Boosting** | 15min | ~80-120 | Engineered indicators + MTF |
| **TCN/LSTM** | 5min | ~40-80 | Base indicators, single-TF |
| **PatchTST/Transformers** | 1min | 4-12 | Raw OHLCV only |

---

## Part 1: Tabular Models (XGBoost, LightGBM, CatBoost)

### Optimal Feature Count

| Sample Size | Recommended Features | Approach |
|-------------|---------------------|----------|
| <10,000 | 30-50 | Aggressive selection |
| 10,000-50,000 | 50-80 | Moderate selection |
| 50,000-200,000 | 80-120 | Light selection |
| >200,000 | 120-180 | Keep most, prune noise |

**Rule of thumb:** 10-100x samples per feature (de Prado recommendation)

### Feature Selection Methods (Ranked)

| Method | Accuracy | Speed | Best For |
|--------|----------|-------|----------|
| **SHAP** | Highest (80-99%) | Slow | Final selection, interpretability |
| **MDA (Permutation)** | High | Medium | Time series with leakage concerns |
| **MDI (Impurity)** | Medium | Fast | Quick iteration |
| **Boruta** | High | Slow | Initial screening |
| **Boruta+SHAP** | Highest | Slowest | Production-grade |

### De Prado's Recommendations (Advances in Financial ML)

1. **MDA over MDI** - MDA uses OOS performance with purged CV
2. **Clustered Feature Importance** - Group correlated features, evaluate at cluster level
3. **Walk-forward selection** - Re-evaluate importance at each training window

### What Features Work Best for Boosting

| Priority | Feature Type | Examples |
|----------|-------------|----------|
| 1 | Momentum | RSI, MACD, ROC |
| 2 | Volatility | ATR, Bollinger Width |
| 3 | Trend | ADX, MA slopes |
| 4 | Volume/Microstructure | OBV, VWAP distance |
| 5 | MTF confluence | Higher-TF trend confirmation |
| 6 | Wavelets | Low-frequency approximations |

### Noise Handling

- **Irrelevant features:** Boosting largely ignores them (assigns low importance)
- **Correlated features:** Minimal performance impact, but importance splits between them
- **Solution:** Use Clustered MDA/MDI or SHAP

---

## Part 2: Sequence Models (LSTM, GRU, TCN)

### Raw vs Engineered Features

**Research finding (2024 Korean Stock Study):**
> "LSTM networks achieve similar performance to XGBoost despite using only raw OHLCV data without any technical indicators."

| Approach | Best For | Feature Count |
|----------|----------|---------------|
| Raw OHLCV | Large datasets, transformers | 4-10 |
| Light engineering | LSTM/GRU moderate data | 30-50 |
| Heavy engineering | Tabular models, limited data | 100-150 |

### Recommended Configuration

**LSTM/GRU:**
```
Primary TF: 5-15 min
Sequence Length: 60-120 timesteps
Features: 40-80 (selected via MDA/SHAP)
Hidden Size: 64-128 (1.5-2x input features)
Dropout: 0.2-0.3
MTF Strategy: Single-TF or MTF indicators (not multi-stream)
```

**TCN:**
```
Primary TF: 5-15 min
Sequence Length: 100-200 timesteps
Features: 60-100 (can handle more than LSTM)
Filters: 32-64 per layer
Dilations: [1, 2, 4, 8, 16, 32]
```

### MTF Strategy for Sequence Models

| Strategy | Architecture | Use Case |
|----------|--------------|----------|
| Single-TF | Standard LSTM/TCN | Baseline, simpler |
| MTF Feature Concat | Features from multiple TFs joined | Works for LSTM/GRU |
| Multi-Stream Encoders | Separate encoder per TF | PatchTST, iTransformer |

### Noise Mitigation

| Strategy | Description |
|----------|-------------|
| LASSO selection | Eliminates unimportant features |
| Dropout (20%) | Reduces overfitting to noise |
| Input noise injection | Regularization effect |
| Dimensionality reduction | PCA to 50 components before LSTM |
| Early stopping | Prevents overfitting |

---

## Part 3: Transformer Models (PatchTST, iTransformer, TFT)

### Key Insight: Transformers Learn Representations

**PatchTST (ICLR 2023):**
> "Patching design naturally retains local semantic information in embeddings... Compared with best Transformer results, PatchTST achieves 21% reduction in MSE."

**Patching reduces the need for engineered features** - the patching mechanism itself is implicit feature extraction.

### Model-Specific Recommendations

| Model | Input Type | Why |
|-------|-----------|-----|
| **PatchTST** | Raw OHLCV (4-5 features) | Patching learns representations; channel-independence |
| **iTransformer** | Raw multivariate (5-10) | Learns variate-wise representations via attention |
| **TFT** | Engineered (50-100) | VSN designed for heterogeneous features |

### Multi-Resolution Strategy

**MTST (AISTATS 2024):**
> "Patch size controls ability to learn patterns at different frequencies: shorter patches for high-frequency, longer for trends."

**Recommended approach:**
- PatchTST: Multi-stream (1m+5m+15m as separate channels)
- iTransformer: Single-TF (inverted attention captures cross-variate)
- TFT: MTF indicators (add features from other TFs)

### Channel Independence vs Mixing

| Strategy | Best When |
|----------|-----------|
| **Channel Independent (CI)** | Limited data, regime drift, weak correlations |
| **Channel Dependent (CD)** | Strong inter-channel dependencies, sufficient data |

**For financial data:** CI (PatchTST) is more robust to regime changes.

---

## Part 4: Heterogeneous Ensemble Feature Strategy

### Research Consensus: Different Features for Different Models

**JMLR 2023 (Unified Theory of Diversity):**
> "There is now a bias/variance/diversity trade-off. Diversity always subtracts from the expected risk."

**Key principles:**
1. Different feature sets → different errors → better ensemble
2. Correlation between base predictions should be < 0.7
3. Meta-learner benefits from diverse base predictions

### Strategy Comparison

| Strategy | Description | Verdict |
|----------|-------------|---------|
| **A: All 180 features** | Same features for all | Suboptimal - reduces diversity |
| **B: Tailored per-model** | Each family gets suited features | **Recommended** |
| **C: Random subspace** | Random feature subsets | Good for homogeneous, not heterogeneous |

### Recommended Feature Allocation

```
1-min Canonical OHLCV
       │
       ├── CatBoost: 15min + MTF indicators
       │   └── ~120 features (indicators + wavelets + MTF)
       │
       ├── TCN: 5min, single-TF
       │   └── ~80 features (base indicators, no MTF)
       │
       └── PatchTST: 1min multi-stream
           └── 12 features (raw OHLCV × 3 streams)
       │
       ▼
   Meta-Learner (Logistic/Ridge)
```

### Meta-Learner Considerations

**Should meta-learner know what features each base used?** **No.**

- Meta-learner operates on **prediction space** only
- Receives OOF probabilities from each base model
- Does not need to know underlying feature sets
- Simple meta-learner (Logistic, Ridge) prevents overfitting

---

## Part 5: Leakage Prevention in Feature Selection

### Critical Rule: Feature Selection Inside CV Folds

```
WRONG:
  1. Select features on full dataset
  2. Run cross-validation

CORRECT:
  For each CV fold:
    1. Select features using ONLY training portion
    2. Transform validation with fitted selector
    3. Train model on selected features
```

### Walk-Forward Feature Selection

```
For each walk-forward window:
  1. Select features using only training window data
  2. Train model on selected features
  3. Validate on forward window
  4. Record which features were selected
  5. Advance window and repeat

Keep features selected in >70% of windows
```

### Purge/Embargo for Feature Selection

Same purge/embargo as model training:
- **Purge:** 60 bars (removes training observations overlapping with test labels)
- **Embargo:** 1440 bars (removes observations after test period)

---

## Part 6: Implementation Recommendations

### Feature Selection Pipeline

```
Step 1: Initial Screening (Boruta + LightGBM)
  Target: 120-140 features

Step 2: Correlation Clustering
  Remove features with >0.8 correlation
  Target: 80-100 features

Step 3: Walk-Forward MDA (with purge+embargo)
  Evaluate importance on rolling windows
  Target: 60-80 features for boosting

Step 4: Per-Model Allocation
  Boosting: 80-120 engineered features
  Sequence: 40-80 features
  Transformer: 4-12 raw features
```

### Validation Checklist

- [ ] Feature selection occurs INSIDE CV folds (no leakage)
- [ ] Walk-forward selection with purge/embargo
- [ ] Feature stability tracking (% folds selected)
- [ ] Prediction correlation < 0.7 between base models
- [ ] Compare tailored vs all-features performance
- [ ] Meta-learner trained only on OOF predictions
- [ ] Final validation on held-out test set

### Your Current Implementation Gap

**Current state:** All models receive same ~180 features by default

**Recommended change:** Implement per-model feature configs:

```python
# config/feature_sets/catboost.yaml
feature_set:
  include_mtf: true
  include_prefixes: [momentum_, volatility_, trend_, volume_, wavelet_, micro_]
  target_count: 100-120

# config/feature_sets/tcn.yaml
feature_set:
  include_mtf: false
  include_prefixes: [momentum_, volatility_, trend_, volume_]
  target_count: 60-80

# config/feature_sets/patchtst.yaml
feature_set:
  raw_ohlcv_only: true
  multi_stream: [1min, 5min, 15min]
  target_count: 12
```

---

## Summary Table

| Model | Features | MTF | Raw/Engineered | Selection Method |
|-------|----------|-----|----------------|------------------|
| **CatBoost** | 80-120 | Yes (indicators) | Engineered | MDA + SHAP |
| **LightGBM** | 80-120 | Yes (indicators) | Engineered | MDA + SHAP |
| **XGBoost** | 80-120 | Yes (indicators) | Engineered | MDA + SHAP |
| **LSTM** | 40-80 | No | Light engineering | MDA |
| **GRU** | 40-80 | No | Light engineering | MDA |
| **TCN** | 60-100 | No | Light engineering | MDA |
| **PatchTST** | 4-12 | Multi-stream | Raw OHLCV | None needed |
| **iTransformer** | 5-10 | No | Raw OHLCV | None needed |
| **TFT** | 50-100 | Yes (indicators) | Engineered | VSN (built-in) |

---

## Key Takeaways

1. **NO:** All models should NOT get 180 features
2. **YES:** Tailored features per model family improves ensemble diversity
3. **Boosting:** 80-120 engineered features with MTF indicators
4. **Sequence:** 40-100 features, single-TF, no MTF
5. **Transformers:** Raw OHLCV (4-12 features), let model learn representations
6. **Feature selection MUST occur inside CV folds** (leakage prevention)
7. **Walk-forward selection** identifies regime-stable features
8. **Meta-learner** operates on predictions only, doesn't need feature knowledge

---

## Sources

### Tabular Models
- [Advances in Financial Machine Learning - de Prado](https://www.amazon.com/Advances-Financial-Machine-Learning-Marcos/dp/1119482089)
- [Clustered MDA/MDI - mlfinlab](https://www.mlfinlab.com/en/latest/feature_importance/clustered.html)
- [Boruta+SHAP Comparison 2024](https://pubmed.ncbi.nlm.nih.gov/38511557/)
- [Walk-Forward Analysis - IBKR](https://www.interactivebrokers.com/campus/ibkr-quant-news/the-future-of-backtesting-a-deep-dive-into-walk-forward-analysis/)

### Sequence Models
- [Stock Prediction with Raw OHLCV 2024](https://arxiv.org/html/2504.02249)
- [Deep Learning Price Forecasting - Wiley 2024](https://wires.onlinelibrary.wiley.com/doi/10.1002/widm.1519)
- [TCN-LSTM with LASSO 2024](https://pubs.acs.org/doi/10.1021/acsomega.3c06263)
- [LSTM Dimensionality Reduction 2024](https://www.sciencedirect.com/science/article/abs/pii/S1568494624001522)

### Transformer Models
- [PatchTST - ICLR 2023](https://arxiv.org/abs/2211.14730)
- [iTransformer - ICLR 2024](https://github.com/thuml/iTransformer)
- [TFT Paper](https://arxiv.org/abs/1912.09363)
- [MTST - AISTATS 2024](https://arxiv.org/abs/2311.04147)
- [Channel Independence Study - TKDE 2024](https://github.com/hanlu-nju/channel_independent_MTSF)

### Heterogeneous Ensembles
- [Unified Theory of Diversity - JMLR 2023](https://jmlr.org/papers/volume24/23-0041/23-0041.pdf)
- [Building Heterogeneous Ensembles - Springer 2021](https://link.springer.com/article/10.1007/s13042-021-01442-1)
- [LLM-TOPLA Diversity - EMNLP 2024](https://aclanthology.org/2024.findings-emnlp.698.pdf)
- [Purged Cross-Validation](https://en.wikipedia.org/wiki/Purged_cross-validation)
- [CPCV - QuantInsti](https://blog.quantinsti.com/cross-validation-embargo-purging-combinatorial/)
