# Best Models for OHLCV Time Series Prediction (2024-2025)

## Comprehensive Research Report

**Last Updated:** December 2024
**Purpose:** State-of-the-art model selection for financial time series prediction
**Focus Areas:** Price direction, volatility forecasting, trading signals, multi-horizon prediction

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Model Rankings Table](#model-rankings-table)
3. [Detailed Model Analysis](#detailed-model-analysis)
   - [Foundation Models](#1-foundation-models)
   - [Transformer Variants](#2-transformer-variants)
   - [State Space Models](#3-state-space-models-mamba-s4)
   - [Temporal Convolutional Networks](#4-temporal-convolutional-networks-tcn)
   - [Gradient Boosting Models](#5-gradient-boosting-models)
   - [Neural Network Models](#6-neural-network-models)
   - [Hybrid Architectures](#7-hybrid-architectures)
   - [Simple Baselines](#8-simple-but-strong-baselines)
4. [Ensemble Recommendations](#ensemble-recommendations)
5. [Implementation Priority](#implementation-priority)
6. [Hyperparameter Guidelines](#hyperparameter-guidelines)
7. [Sources](#sources)

---

## Executive Summary

Based on extensive research of 2024-2025 publications, benchmarks, and production deployments, the landscape of OHLCV time series prediction has evolved significantly. Key findings:

### Top Performers by Category

| Category | Best Models | Key Insight |
|----------|-------------|-------------|
| **Zero-Shot Forecasting** | Chronos-Bolt, TimesFM 2.5 | Foundation models now competitive with trained models |
| **Long-Horizon Forecasting** | PatchTST, iTransformer | Transformer variants dominate long-term prediction |
| **Short-Term Trading** | XGBoost, LightGBM, CatBoost | Gradient boosting still king for tabular financial data |
| **Efficiency** | TiDE, DLinear | Simple models often match complex ones at 10x speed |
| **Hybrid Long-Short** | Mamba-Transformer (SST, MoU) | Combining SSM efficiency with Transformer expressiveness |

### Critical Finding for Financial Data

**Ensemble gradient boosting (XGBoost, LightGBM, CatBoost) consistently outperforms neural approaches for daily financial returns prediction when trained on relevant features.** Time series foundation models perform weakly in zero-shot financial forecasting but improve significantly when combined with financial factors.

---

## Model Rankings Table

### Tier 1: Production-Ready (Recommended for Implementation)

| Rank | Model | Accuracy | Speed | Memory | Interpretability | Best For |
|------|-------|----------|-------|--------|------------------|----------|
| 1 | **XGBoost/LightGBM/CatBoost** | 9/10 | 10/10 | 9/10 | 7/10 | Tabular features, multi-horizon |
| 2 | **PatchTST** | 8/10 | 7/10 | 7/10 | 5/10 | Long-term forecasting |
| 3 | **iTransformer** | 8/10 | 7/10 | 6/10 | 5/10 | Multivariate correlations |
| 4 | **TCN** | 8/10 | 9/10 | 8/10 | 6/10 | Sequence patterns |
| 5 | **TiDE** | 7/10 | 10/10 | 9/10 | 6/10 | Fast production inference |
| 6 | **Chronos-Bolt** | 7/10 | 8/10 | 7/10 | 4/10 | Zero-shot, no training needed |

### Tier 2: Strong Alternatives

| Rank | Model | Accuracy | Speed | Memory | Interpretability | Best For |
|------|-------|----------|-------|--------|------------------|----------|
| 7 | **S-Mamba** | 7/10 | 9/10 | 9/10 | 4/10 | Long sequences, linear scaling |
| 8 | **TimesFM 2.5** | 7/10 | 8/10 | 7/10 | 4/10 | Zero-shot foundation model |
| 9 | **N-BEATS/N-HiTS** | 7/10 | 8/10 | 7/10 | 7/10 | Interpretable decomposition |
| 10 | **TimesNet** | 7/10 | 6/10 | 6/10 | 5/10 | Multi-task learning |
| 11 | **Autoformer** | 7/10 | 6/10 | 6/10 | 5/10 | Seasonal patterns |
| 12 | **CNN-LSTM** | 7/10 | 7/10 | 7/10 | 5/10 | Hybrid spatial-temporal |

### Tier 3: Specialized Use Cases

| Rank | Model | Accuracy | Speed | Memory | Interpretability | Best For |
|------|-------|----------|-------|--------|------------------|----------|
| 13 | **GNN-LSTM** | 6/10 | 5/10 | 5/10 | 4/10 | Relational stock data |
| 14 | **DLinear/NLinear** | 6/10 | 10/10 | 10/10 | 9/10 | Baseline, trend extraction |
| 15 | **Informer** | 6/10 | 7/10 | 6/10 | 4/10 | Very long sequences |
| 16 | **FinGPT** | 5/10 | 3/10 | 3/10 | 6/10 | Sentiment + price fusion |

---

## Detailed Model Analysis

### 1. Foundation Models

#### Chronos (Amazon)
- **Paper:** "Chronos: Learning the Language of Time Series" (2024)
- **Architecture:** T5-based encoder-decoder, tokenizes time series values
- **Key Innovation:** Treats time series as a language modeling problem
- **Versions:**
  - Chronos (Original): 20M-710M parameters
  - Chronos-Bolt: 250x faster, 20x more memory efficient
  - Chronos-2: 120M params, supports multivariate + covariates

**Performance:**
- Zero-shot: Outperforms DeepAR and GPT-3-based forecasting by >25%
- Chronos-Bolt (Base) surpasses original Chronos (Large) while being 600x faster
- 600M+ downloads on HuggingFace

**Financial Application:**
- Combined with financial factors: Chronos (small) achieved 51.74% directional accuracy
- Annualized return: 41.89%, Sharpe: 6.78 (vs CatBoost: 47.25% return, 6.46 Sharpe)

**Best Configuration:**
```python
from chronos import ChronosPipeline
pipeline = ChronosPipeline.from_pretrained("amazon/chronos-bolt-base")
# Context: up to 2048 points for v2.0+
# Output: Probabilistic forecasts with quantiles
```

#### TimesFM (Google)
- **Paper:** "A Decoder-only Foundation Model for Time Series Forecasting" (ICML 2024)
- **Architecture:** Decoder-only transformer, 200M parameters
- **Key Innovation:** Patching strategy from PatchTST + decoder-only design

**Versions:**
- TimesFM 1.0-200m: Up to 512 context, point forecasts only
- TimesFM 2.0-500m: Up to 2048 context, 25% better than v1.0
- TimesFM 2.5-200m: Latest release

**Performance:**
- Zero-shot performance comparable to supervised SOTA
- Best overall RMSE (0.3023) in comparative studies
- Available in Google Cloud BigQuery

**Best Configuration:**
```python
import timesfm
model = timesfm.TimesFM(
    context_length=512,
    horizon_length=96,
    freq=0  # 0=high freq, 1=medium, 2=low
)
```

### 2. Transformer Variants

#### PatchTST
- **Paper:** "A Time Series is Worth 64 Words" (ICLR 2023)
- **Architecture:** Channel-independent patching + Transformer
- **Key Innovation:** Segments time series into patches as tokens

**Performance:**
- 21% MSE reduction vs previous Transformer SOTA
- VMD-PatchTST-ASWL framework shows strong stock index prediction
- Particularly well-suited for stock price prediction

**Why It Works for OHLCV:**
1. Patching captures local semantic information (candlestick patterns)
2. Channel-independence handles multiple features (OHLCV) separately
3. Quadratic attention cost reduced by patch aggregation

**Best Configuration:**
```python
# Recommended hyperparameters for financial data
patch_length = 16  # or 64
stride = 8
d_model = 256
n_heads = 8
e_layers = 3
d_ff = 512
context_length = 512  # lookback window
```

#### iTransformer
- **Paper:** "Inverted Transformers Are Effective for Time Series Forecasting" (ICLR 2024 Spotlight)
- **Architecture:** Inverted attention (variates as tokens instead of time steps)
- **Key Innovation:** Attention captures multivariate correlations

**Performance:**
- State-of-the-art on multivariate forecasting benchmarks
- Can forecast unseen variates with good generalization
- Better utilization of arbitrary lookback windows

**Why It Works for OHLCV:**
1. Treats each feature (O, H, L, C, V) as a token
2. Attention learns relationships between price levels
3. Handles varying lookback windows gracefully

**Best Configuration:**
```python
pip install iTransformer
# Key params:
d_model = 512
n_heads = 8
e_layers = 4
seq_len = 96
pred_len = 24
```

#### TimesNet
- **Paper:** "Temporal 2D-Variation Modeling for General Time Series Analysis" (ICLR 2023)
- **Architecture:** Transforms 1D time series to 2D tensors based on periodicity
- **Key Innovation:** 2D convolutions capture intra/inter-period variations

**Performance:**
- SOTA across 5 tasks: forecasting, imputation, classification, anomaly detection
- Multi-task capabilities useful for trading systems
- Not yet proven superior for volatility prediction specifically

**Why It Works for OHLCV:**
- Multi-periodicity discovery (daily, weekly, monthly patterns)
- 2D kernels model complex temporal variations

#### Autoformer
- **Paper:** "Autoformer: Decomposition Transformers with Auto-Correlation"
- **Architecture:** Decomposition + auto-correlation mechanism
- **Key Innovation:** Seasonal-trend decomposition built into transformer

**Financial Performance:**
- Highest Sharpe and Calmar ratios among Transformer variants in financial tests
- Most practical for financial markets in comparative studies
- Better for seasonal pattern recognition

### 3. State Space Models (Mamba, S4)

#### S-Mamba (Simple-Mamba)
- **Key Innovation:** Selective state space with near-linear complexity
- **Performance:** Tested on 13 datasets (traffic, electricity, weather, finance, energy)

**Advantages over Transformers:**
- Linear O(L) complexity vs quadratic O(L^2)
- 78% fewer parameters than iTransformer
- Low GPU memory and fast training

**Financial Results:**
- Bi-Mamba4TS: 4.92% MSE reduction, 2.16% MAE reduction vs Transformers

#### Hybrid Mamba-Transformer (SST, MoU)

**SST (State Space Transformer):**
- Mamba expert for long-range patterns
- Transformer expert for short-term dynamics
- Linear scaling O(L) maintained

**MoU (Mixture of Universals):**
- Hierarchically integrates Mamba, FFN, Convolution, Self-Attention
- Most versatile architecture for varying pattern types

**Best Configuration:**
```python
# SST multi-scale approach
scales = [1, 4, 16]  # Multi-scale time aggregation
mamba_layers = 2    # Long-range
transformer_layers = 2  # Short-range
```

### 4. Temporal Convolutional Networks (TCN)

#### Core TCN Architecture
- **Key Innovation:** Dilated causal convolutions for exponential receptive field
- **WaveNet Variant:** Residual connections + dilations + causal convolutions

**Performance vs LSTM:**
- 2x faster training than LSTM
- Slightly better accuracy in most benchmarks
- More stable training, lower memory requirements

**Financial Results:**
- Heating oil: 53% accuracy, 0.005 edge
- RBOB gasoline: 52.2% accuracy, 0.001 edge
- TCN-LSTM hybrid: Higher robustness for nonlinear/unstable data

**Best Configuration:**
```python
# TCN for OHLCV
kernel_size = 3
dilations = [1, 2, 4, 8, 16, 32]  # Exponential growth
num_channels = [64, 64, 64, 64, 64, 64]
dropout = 0.2
# Receptive field = 1 + 2*(kernel_size-1)*sum(dilations) = 253 timesteps
```

#### ACTGAN (Attentive TCN + GAN)
- **Architecture:** GAN framework with attentive TCN generator
- **Performance:** 78.29% MAE reduction vs ARIMA, 51.01% vs GRU

### 5. Gradient Boosting Models

#### XGBoost, LightGBM, CatBoost

**Critical Finding:** These models consistently outperform neural networks and foundation models for daily financial returns when given proper features.

**Comparative Performance (2024 Study, 2001-2023):**

| Model | Directional Accuracy | Best Use Case |
|-------|---------------------|---------------|
| CatBoost | 51.16% | General, handles categoricals |
| LightGBM | ~51% | Large datasets, speed |
| XGBoost | ~51% | Regularization, robustness |

**Key Insight:** Off-the-shelf pre-trained TSFMs perform weakly in zero-shot financial forecasting, underperforming CatBoost and LightGBM.

**Recommended Configuration:**

```python
# XGBoost for multi-horizon prediction
import xgboost as xgb

params = {
    'max_depth': 6,
    'learning_rate': 0.05,
    'n_estimators': 500,
    'min_child_weight': 3,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,  # L1 regularization
    'reg_lambda': 1.0,  # L2 regularization
    'tree_method': 'hist',  # Faster
    'objective': 'binary:logistic'  # For direction
}

# LightGBM (fastest)
import lightgbm as lgb

params = {
    'num_leaves': 31,
    'max_depth': -1,
    'learning_rate': 0.05,
    'n_estimators': 500,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0
}

# CatBoost (best for categoricals)
from catboost import CatBoostClassifier

model = CatBoostClassifier(
    iterations=500,
    depth=6,
    learning_rate=0.05,
    l2_leaf_reg=3.0,
    random_strength=1.0,
    bagging_temperature=1.0
)
```

### 6. Neural Network Models

#### N-BEATS (Neural Basis Expansion Analysis)
- **Architecture:** Deep stacks of fully connected layers with basis expansion
- **Key Innovation:** Trend + seasonality decomposition via learned basis functions
- **Performance:** 11% improvement over statistical benchmarks

**Best Configuration:**
```python
from neuralforecast.models import NBEATS

model = NBEATS(
    input_size=30,           # lookback
    h=5,                     # forecast horizon
    stack_types=['trend', 'seasonality'],
    n_blocks=[3, 3],
    mlp_units=[[512, 512], [512, 512]],
    num_harmonics=4,         # For seasonality
    polynomial_degree=3      # For trend
)
```

#### N-HiTS (Neural Hierarchical Interpolation)
- **Extension of N-BEATS with hierarchical interpolation**
- **Better for multi-horizon forecasting**

#### CNN-LSTM Hybrid
- **Architecture:** CNN for spatial features + LSTM for temporal
- **Performance:** 15% RMSE improvement over standalone models
- **Best for:** When both pattern recognition and sequence memory needed

**Best Configuration:**
```python
# Hybrid CNN-LSTM for OHLCV
model = Sequential([
    # CNN feature extraction
    Conv1D(filters=64, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Conv1D(filters=128, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),

    # LSTM sequence modeling
    LSTM(units=100, return_sequences=True),
    LSTM(units=50),

    # Output
    Dense(units=1, activation='sigmoid')
])
```

#### GNN + LSTM (Graph Neural Networks)
- **Use Case:** When stock relationships matter (sector correlations, ETF constituents)
- **Performance:** TFT-GNN outperformed standalone TFT in 11/12 periods
- **Challenge:** Generating effective relationship graphs is difficult

### 7. Hybrid Architectures

#### Recommended Hybrid Combinations

| Hybrid | Components | Benefit | Use Case |
|--------|------------|---------|----------|
| **ARIMA + CNN-LSTM + XGBoost** | Statistical + Deep + Boosting | Captures linear + nonlinear | General forecasting |
| **VMD + PatchTST + ASWL** | Decomposition + Transformer | Multi-scale patterns | Stock indices |
| **TCN-LSTM** | Conv + Recurrent | Efficiency + Memory | Real-time prediction |
| **Mamba-Transformer (SST)** | SSM + Attention | Long + Short patterns | Any sequence length |
| **TFT-GNN** | Temporal + Graph | Time + Relationships | Portfolio prediction |

### 8. Simple but Strong Baselines

#### DLinear
- **Architecture:** Trend-seasonality decomposition + two linear layers
- **Performance:** Outperforms FEDformer by 25-40% on various datasets

**When to Use:**
- When trend is the dominant signal
- As a baseline to beat
- For extremely fast inference

#### NLinear
- **Architecture:** Simple normalization + linear layer
- **When to Use:** When distribution shift is a problem

#### TiDE (Time-series Dense Encoder)
- **Architecture:** MLP encoder-decoder
- **Performance:** 5-10x faster than Transformers with comparable accuracy
- **Production Ready:** Available in Google Cloud Vertex AI

**Best Configuration:**
```python
# TiDE configuration
hidden_size = 256
encoder_layers = 2
decoder_layers = 2
temporal_hidden_size = 32
dropout = 0.3
```

---

## Ensemble Recommendations

### Best 3-Model Ensemble (Recommended Starting Point)

```
Ensemble = 0.4 * XGBoost + 0.3 * LightGBM + 0.3 * CatBoost
```

**Why This Works:**
1. Model diversity: Different tree-building algorithms
2. Complementary errors: XGBoost (regularized), LightGBM (leaf-wise), CatBoost (ordered boosting)
3. Fast training and inference
4. No need for feature engineering differences

### Best 5-Model Ensemble (Production Grade)

```
Tier 1 (Tabular): XGBoost + LightGBM + CatBoost
Tier 2 (Sequential): TCN + PatchTST (or iTransformer)
Meta-Learner: Ridge Regression or small neural network
```

**Stacking Architecture:**
```python
# Level 1: Base models (out-of-fold predictions)
base_models = [
    XGBoostClassifier(),    # Tabular
    LightGBMClassifier(),   # Tabular
    CatBoostClassifier(),   # Tabular (handles categoricals)
    TCNModel(),             # Sequential
    PatchTSTModel()         # Transformer
]

# Level 2: Meta-learner
meta_learner = RidgeClassifier()  # or LogisticRegression

# Critical: Use out-of-fold predictions for meta-learner training
```

### Diversity Metrics

For effective ensembling, ensure base models have:
1. **Correlation < 0.7** between predictions
2. **Different error patterns**: Some overpredict, some underpredict
3. **Architecture diversity**: Mix tabular + sequential + attention models

### Blending vs Stacking

| Method | When to Use | Advantage |
|--------|-------------|-----------|
| **Simple Average** | Models have similar accuracy | Robust, no overfitting |
| **Weighted Average** | One model clearly better | Leverage strength |
| **Stacking** | Large validation set available | Learns optimal combination |
| **Blending** | Limited data | Uses holdout, faster |

---

## Implementation Priority

### Phase 1: Baseline Models (Week 1-2)

| Priority | Model | Reason | Expected Effort |
|----------|-------|--------|-----------------|
| 1 | XGBoost | Best tabular performance, well-documented | 2 days |
| 2 | LightGBM | Fastest training, production-proven | 1 day |
| 3 | CatBoost | Handles categoricals, robust | 1 day |
| 4 | DLinear | Simple baseline to beat | 0.5 days |

### Phase 2: Sequential Models (Week 3-4)

| Priority | Model | Reason | Expected Effort |
|----------|-------|--------|-----------------|
| 5 | TCN | Fast, proven for financial data | 2 days |
| 6 | LSTM | Standard benchmark | 1 day |
| 7 | N-BEATS | Interpretable, good performance | 2 days |

### Phase 3: Transformer Models (Week 5-6)

| Priority | Model | Reason | Expected Effort |
|----------|-------|--------|-----------------|
| 8 | PatchTST | Current SOTA for long-term | 3 days |
| 9 | iTransformer | Multivariate correlations | 2 days |
| 10 | TFT | Known good for finance | 3 days |

### Phase 4: Foundation Models (Week 7-8)

| Priority | Model | Reason | Expected Effort |
|----------|-------|--------|-----------------|
| 11 | Chronos-Bolt | Zero-shot baseline | 1 day |
| 12 | TimesFM | Google foundation model | 1 day |

### Phase 5: Advanced/Experimental (Week 9+)

| Priority | Model | Reason | Expected Effort |
|----------|-------|--------|-----------------|
| 13 | Mamba/S-Mamba | Efficiency, long sequences | 3 days |
| 14 | SST/MoU | Hybrid Mamba-Transformer | 4 days |
| 15 | GNN-LSTM | Relational data | 5 days |

---

## Hyperparameter Guidelines

### Universal Hyperparameters for Financial Time Series

```python
# Lookback windows (OHLCV at 5-minute bars)
SHORT_LOOKBACK = 12 * 6   # 6 hours = 72 bars
MEDIUM_LOOKBACK = 12 * 24  # 1 day = 288 bars
LONG_LOOKBACK = 12 * 24 * 5  # 1 week = 1440 bars

# Forecast horizons
HORIZONS = [5, 10, 15, 20]  # Match your label horizons

# Training
BATCH_SIZE = 64  # or 128 for larger datasets
LEARNING_RATE = 1e-4  # Transformers: 1e-4, MLPs: 1e-3
DROPOUT = 0.1  # 0.1-0.3 for finance
WEIGHT_DECAY = 1e-5  # Regularization

# Early stopping
PATIENCE = 10  # epochs
MIN_DELTA = 1e-4
```

### Model-Specific Hyperparameters

#### Gradient Boosting (XGBoost/LightGBM/CatBoost)
```python
# Optuna search space
SEARCH_SPACE = {
    'max_depth': [4, 6, 8, 10],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 300, 500, 1000],
    'min_child_weight': [1, 3, 5, 7],
    'subsample': [0.6, 0.7, 0.8, 0.9],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
    'reg_alpha': [0, 0.1, 0.5, 1.0],
    'reg_lambda': [0.1, 0.5, 1.0, 2.0]
}
```

#### PatchTST
```python
PATCHTST_CONFIG = {
    'context_length': 512,
    'patch_length': 16,  # or 64
    'stride': 8,
    'd_model': 256,
    'n_heads': 8,
    'e_layers': 3,
    'd_ff': 512,
    'dropout': 0.1,
    'learning_rate': 1e-4
}
```

#### iTransformer
```python
ITRANSFORMER_CONFIG = {
    'seq_len': 96,
    'pred_len': 24,
    'd_model': 512,
    'n_heads': 8,
    'e_layers': 4,
    'd_ff': 2048,
    'dropout': 0.1
}
```

#### TCN
```python
TCN_CONFIG = {
    'input_size': 5,  # OHLCV
    'output_size': 1,
    'num_channels': [64] * 6,
    'kernel_size': 3,
    'dropout': 0.2,
    'dilations': [1, 2, 4, 8, 16, 32]
}
```

---

## Sources

### Foundation Models
- [Chronos GitHub](https://github.com/amazon-science/chronos-forecasting)
- [TimesFM - Google Research](https://research.google/blog/a-decoder-only-foundation-model-for-time-series-forecasting/)
- [TimesFM on HuggingFace](https://huggingface.co/google/timesfm-2.5-200m-pytorch)
- [Chronos-2 Announcement](https://www.amazon.science/blog/introducing-chronos-2-from-univariate-to-universal-forecasting)

### Transformer Models
- [PatchTST Paper](https://arxiv.org/abs/2211.14730)
- [PatchTST GitHub](https://github.com/yuqinie98/PatchTST)
- [iTransformer Paper](https://arxiv.org/abs/2310.06625)
- [iTransformer GitHub](https://github.com/thuml/iTransformer)
- [TimesNet Paper](https://arxiv.org/abs/2210.02186)
- [TimesNet GitHub](https://github.com/thuml/TimesNet)
- [Informer Paper](https://arxiv.org/abs/2012.07436)
- [Autoformer - HuggingFace Blog](https://huggingface.co/blog/autoformer)

### State Space Models
- [Mamba Effectiveness for TSF](https://arxiv.org/abs/2403.11144)
- [SST: Mamba-Transformer Hybrid](https://arxiv.org/abs/2404.14757)
- [MoU: Mixture of Universals](https://arxiv.org/abs/2408.15997)
- [S4 Annotated](https://srush.github.io/annotated-s4/)

### Temporal Convolutional Networks
- [TCN Unit8 Guide](https://unit8.com/resources/temporal-convolutional-networks-and-forecasting/)
- [TCN vs LSTM Comparison - Kaggle](https://www.kaggle.com/code/ricardocolindres/lstm-vs-tcn-for-time-series-analysis-comparison)
- [WaveNet Financial Application](https://bora.uib.no/bora-xmlui/handle/11250/3144546)
- [ACTGAN Paper](https://www.sciencedirect.com/science/article/pii/S2590005625000013)

### Gradient Boosting
- [XGBoost vs LightGBM vs CatBoost - Skforecast](https://cienciadedatos.net/documentos/py39-forecasting-time-series-with-skforecast-xgboost-lightgbm-catboost)
- [Foundation Models vs Boosting in Finance](https://arxiv.org/html/2511.18578v1)

### Ensemble Methods
- [Stacking Ensemble for Forecasting](https://cienciadedatos.net/documentos/py52-stacking-ensemble-models-forecasting.html)
- [Multi-layer Stack Ensembles](https://arxiv.org/abs/2511.15350)
- [Blending for Crude Oil Forecasting](https://link.springer.com/article/10.1007/s10479-023-05810-8)

### Neural Networks
- [N-BEATS Paper](https://arxiv.org/abs/1905.10437)
- [N-HiTS Financial Comparison](https://arxiv.org/abs/2409.00480)
- [TiDE - Google Research](https://research.google/pubs/long-horizon-forecasting-with-tide-time-series-dense-encoder/)
- [DLinear GitHub](https://github.com/cure-lab/LTSF-Linear)

### Hybrid Models
- [CNN-LSTM for Stock Prediction](https://arxiv.org/html/2502.15813v1)
- [GNN for Stock Prediction Survey](https://dl.acm.org/doi/10.1145/3696411)
- [TFT-GNN Hybrid](https://www.mdpi.com/2673-9909/5/4/176)

### Financial LLMs
- [FinGPT GitHub](https://github.com/AI4Finance-Foundation/FinGPT)
- [BloombergGPT Paper](https://arxiv.org/abs/2303.17564)
- [LLMs for Financial Time Series](https://arxiv.org/abs/2306.11025)

### Benchmarks and Reviews
- [Deep Learning for Financial Time Series - Survey](https://www.sciencedirect.com/org/science/article/pii/S152614922300125X)
- [Transformer TSF Hyperparameter Pipeline](https://arxiv.org/html/2501.01394)
- [Time Series Transformers Review - GitHub](https://github.com/qingsongedu/time-series-transformers-review)

---

## Summary: What to Implement First

For OHLCV prediction with your existing Phase 1 pipeline (150+ features, multi-horizon labels, quality weights):

1. **Start with XGBoost/LightGBM/CatBoost ensemble** - These will likely be your production models
2. **Add TCN as sequence baseline** - Captures temporal patterns efficiently
3. **Add PatchTST for long-horizon** - State-of-the-art Transformer for time series
4. **Consider Chronos-Bolt** - Zero-shot baseline, no training required
5. **Ensemble the best 3-5 models** - Stacking with Ridge meta-learner

The research clearly shows that for financial data with engineered features, gradient boosting models remain the most reliable choice, but hybrid approaches combining boosting with sequence models often yield the best ensemble results.
