# Feature Requirements by Model Family

**Research Date:** 2024-12-24
**Scope:** Feature engineering requirements for OHLCV futures prediction
**Pipeline Context:** ~150-200 features across price, momentum, volatility, volume, trend, temporal, regime, microstructure, wavelets, and multi-timeframe categories

---

## Executive Summary

| Model Family | Feature Scaling | Optimal Feature Count | Best Feature Types | Avoid | Key Notes |
|--------------|-----------------|----------------------|-------------------|-------|-----------|
| **Boosting (XGBoost, LightGBM, CatBoost)** | Not required | 20-100 (after selection) | All types work; momentum, volatility, volume most predictive | Highly correlated redundant features | Native handling of missing values; CatBoost handles categoricals natively |
| **LSTM/GRU** | Required (MinMax [0,1] or StandardScaler) | 10-50 per timestep | Price ratios, returns, normalized indicators | Raw prices, unbounded features | Wavelets highly beneficial; sequence length 30-100 steps |
| **Transformers (PatchTST, TimesFM)** | Varies by model | Can use raw OHLCV or engineered | Patched raw data or minimal features | Over-engineering may hurt | Zero-shot capable; fine-tuning improves domain fit |
| **Random Forest** | Not required | 20-80 (after selection) | All types; handles high dimensionality | Highly correlated features reduce diversity | Feature importance for selection |
| **SVM** | Required (StandardScaler preferred) | 10-50 | Normalized indicators, bounded features | Unbounded, high-dimensional sparse | RBF kernel most common |
| **Logistic Regression** | Required (StandardScaler) | 10-30 | Stationary, normalized features | Non-linear relationships | Regularization (L1/L2) for selection |
| **Ensemble Meta-Learners** | Depends on meta-model | Base predictions + select originals | Base model predictions | Duplicating base model features | Cross-validated predictions critical |

---

## 1. Boosting Models (XGBoost, LightGBM, CatBoost)

### 1.1 Feature Scaling Requirements

**Scaling NOT required.** Tree-based models are invariant to monotonic transformations of features because splits are based on rank ordering, not absolute values.

> "Random Forest is a tree-based model and hence does not require feature scaling. Tree-based models are invariant to the scale of the features." - [Forecastegy](https://forecastegy.com/posts/does-random-forest-need-feature-scaling-or-normalization/)

> "Ensemble methods (such as Random Forest and gradient boosting models like XGBoost, CatBoost and LightGBM) demonstrate robust performance largely independent of scaling." - [arXiv](https://arxiv.org/html/2506.08274v1)

### 1.2 Optimal Feature Count

- **Initial:** 100-200 features acceptable
- **After selection:** 20-100 features optimal
- **Methods:**
  - XGBoost/LightGBM gain importance
  - SHAP values for interpretability
  - Recursive Feature Elimination (RFE)
  - Boruta algorithm (wraps Random Forest)
  - BoostARoota (XGBoost-native)

> "The Boruta algorithm can effectively reduce the data dimension and noise and improve the model's capacity for generalization by automatically identifying and screening features highly correlated with target variables." - [MDPI Systems](https://www.mdpi.com/2079-8954/12/7/254)

### 1.3 Best Feature Types for This Pipeline

| Category | Effectiveness | Notes |
|----------|--------------|-------|
| **Momentum (RSI, MACD, ROC)** | Excellent | Captures mean-reversion and trend signals |
| **Volatility (ATR, Bollinger, Parkinson)** | Excellent | Regime identification, risk signals |
| **Volume (OBV, VWAP, dollar volume)** | Good | Confirms price movements |
| **Multi-timeframe (MTF)** | Excellent | Multiple perspectives on same asset |
| **Wavelets** | Good | Frequency-domain decomposition |
| **Temporal (hour, session flags)** | Good | Market microstructure patterns |
| **Microstructure proxies** | Good | Liquidity and spread signals |
| **Raw prices** | Poor | Use returns/ratios instead |

### 1.4 Categorical Feature Handling

**XGBoost:** Requires manual encoding (label or one-hot)

**LightGBM:** Native categorical support via `categorical_feature` parameter
> "LightGBM can handle categorical features by taking the input of feature names without converting to one-hot coding, making it much faster." - [KDnuggets](https://www.kdnuggets.com/2018/03/catboost-vs-light-gbm-vs-xgboost.html)

**CatBoost:** Best native categorical handling using ordered target encoding
> "CatBoost uses a variant of target encoding called 'ordered encoding' to avoid target leakage. This approach mimics time series data validation and helps prevent overfitting." - [DataCamp](https://www.datacamp.com/tutorial/catboost)

### 1.5 Feature Interaction Handling

- All three handle interactions automatically through tree splits
- Consider explicit interaction features for known domain relationships
- Polynomial features generally not needed

### 1.6 Specific Recommendations for This Pipeline

```python
# Feature selection using XGBoost importance
from xgboost import XGBClassifier
from sklearn.feature_selection import SelectFromModel

model = XGBClassifier(n_estimators=100, importance_type='gain')
model.fit(X_train, y_train)

# Select features above median importance
selector = SelectFromModel(model, threshold='median', prefit=True)
X_selected = selector.transform(X_train)

# For interpretability, use SHAP
import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train)
```

**Sources:**
- [Forecasting with XGBoost, LightGBM, CatBoost](https://cienciadedatos.net/documentos/py39-forecasting-time-series-with-skforecast-xgboost-lightgbm-catboost.html)
- [Real-time XGBoost vs CatBoost](https://emergentmethods.medium.com/real-time-head-to-head-adaptive-modeling-of-financial-market-data-using-xgboost-and-catboost-995a115a7495)
- [LightGBM Feature Importance](https://mljourney.com/lightgbm-feature-importance-comprehensive-guide/)

---

## 2. Neural Networks (LSTM, GRU, Temporal CNN)

### 2.1 Feature Scaling Requirements

**Scaling REQUIRED.** Neural networks use gradient-based optimization that is sensitive to feature magnitudes.

> "Different features may have vastly different scales, which can confuse the LSTM model. That's why scaling is essential." - [Neural Brain Works](https://neuralbrainworks.com/lstm-data-preprocessing-techniques-for-time-series-models/)

**Recommended approaches:**
- **MinMax [0,1]:** Preferred for bounded data, preserves zero
- **StandardScaler:** For unbounded data, better for outliers
- **RobustScaler:** When outliers are present (financial data)

> "For time series forecasting, Min-Max scaling is often preferred, especially when data has known boundaries." - [Machine Learning Mastery](https://machinelearningmastery.com/how-to-scale-data-for-long-short-term-memory-networks-in-python/)

**Critical:** Fit scaler on training data only, transform validation/test with same parameters.

### 2.2 Sequence Length Requirements

| Use Case | Recommended Length | Notes |
|----------|-------------------|-------|
| Short-term (5-20 bars) | 30-60 timesteps | Captures recent momentum |
| Medium-term | 60-120 timesteps | Captures regime changes |
| Long-term patterns | 120-500 timesteps | Consider attention mechanisms |

> "While RNNs, LSTM, and GRU models have significantly improved sequential data processing, they still face challenges when dealing with extremely long sequences." - [Medium](https://medium.com/@jawad.ravian41/understanding-lstm-gru-and-rnn-architectures-and-overcoming-sequence-length-limitations-1b43a193894f)

### 2.3 Best Feature Types for Sequences

| Category | Effectiveness | Notes |
|----------|--------------|-------|
| **Returns (log/simple)** | Excellent | Stationary, normalized |
| **Normalized indicators (RSI, etc.)** | Excellent | Bounded [0,1] or [-1,1] |
| **Volatility ratios** | Excellent | Regime signals |
| **Volume ratios** | Good | Relative to rolling average |
| **Wavelets** | Excellent | Multi-resolution analysis |
| **Raw OHLCV** | Good with scaling | Recent research shows LSTMs can learn from raw data |
| **Unbounded features** | Poor | Must normalize first |

### 2.4 Wavelets for Neural Networks

**Highly beneficial.** Wavelets provide multi-resolution decomposition that helps LSTMs separate signal from noise.

> "A novel algorithm combining wavelet decomposition with LSTM networks achieved prediction accuracy gains of 18.5%, 32.8%, and 36.47% compared to LSTM, RNN, and ANN models respectively." - [Neural Computing and Applications](https://link.springer.com/article/10.1007/s00521-024-10561-z)

> "The proposed WaveFLSTM has advantage of denoising through maximal overlap discrete wavelet transform (MODWT) and integrates fuzzy logic, addressing fuzzy, nonlinear, and non-stationary characteristics of time series." - [Neural Computing and Applications](https://link.springer.com/article/10.1007/s00521-024-10622-3)

**Implementation options:**
1. **Pre-decomposition:** Feed wavelet coefficients as separate features
2. **Parallel processing:** Separate LSTM for each frequency band
3. **Feature enrichment:** Add wavelet energy/volatility as additional features

### 2.5 Temporal Encoding Considerations

- **Cyclical encoding (sin/cos):** Essential for hour, day-of-week, month
- **Session flags:** Binary indicators for market sessions
- **Holiday/event flags:** Binary indicators for known events

### 2.6 LSTM vs GRU Choice

> "GRU is a simpler and computationally more efficient variation of LSTM. The GRU architecture simplifies the LSTM design, making it computationally less expensive, and it often performs comparably well in practice." - [Towards Data Science](https://towardsdatascience.com/building-rnn-lstm-and-gru-for-time-series-using-pytorch-a46e5b094e7b/)

**Use GRU when:** Faster training needed, simpler patterns
**Use LSTM when:** Complex long-term dependencies, larger datasets

### 2.7 Specific Recommendations for This Pipeline

```python
import torch
import torch.nn as nn
from sklearn.preprocessing import RobustScaler

# Use RobustScaler for financial data (handles outliers)
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Feature selection: Use only bounded/normalized features
selected_features = [
    # Returns and ratios
    'return_1', 'return_5', 'return_10',
    'high_low_ratio', 'close_open_ratio',
    # Normalized indicators (already [0,1] or [-1,1])
    'rsi_14', 'stoch_k', 'stoch_d', 'williams_r',
    # Volatility ratios
    'atr_ratio', 'bb_position', 'keltner_position',
    # Wavelet features
    'wavelet_d1_energy', 'wavelet_d2_energy', 'wavelet_approx_trend',
    # Temporal (already sin/cos encoded)
    'hour_sin', 'hour_cos', 'minute_sin', 'minute_cos'
]

# Sequence preparation
sequence_length = 60  # 5 hours of 5-min bars
```

**Sources:**
- [LSTM Data Preprocessing](https://neuralbrainworks.com/lstm-data-preprocessing-techniques-for-time-series-models/)
- [Wavelet-LSTM Approach](https://link.springer.com/article/10.1007/s00521-024-10561-z)
- [Building RNN/LSTM/GRU in PyTorch](https://towardsdatascience.com/building-rnn-lstm-and-gru-for-time-series-using-pytorch-a46e5b094e7b/)

---

## 3. Transformer Models (PatchTST, TimesNet, TimesFM)

### 3.1 Raw OHLCV vs Engineered Features

**Key finding from 2024-2025 research:** Modern transformers can learn directly from raw OHLCV data, sometimes matching or exceeding models with hand-crafted features.

> "Deep learning models trained on raw OHLCV data can achieve comparable performance to traditional ML models using technical indicators. A simple LSTM network trained on raw OHLCV data alone can match the performance of sophisticated ML models that incorporate technical indicators." - [arXiv (Korean Markets Study)](https://arxiv.org/abs/2504.02249)

> "Using full OHLCV data provides better predictive accuracy compared to using only close price or close price with volume. These findings challenge conventional approaches to feature engineering in financial forecasting." - [arXiv](https://arxiv.org/abs/2504.02249)

**However, enhanced features can still help:**
> "BiLSTM, in association with the enhanced dataset, averagely reduce the MSE by 68.44% and the MAE by 47.78% compared to raw data alone." - [Springer](https://link.springer.com/chapter/10.1007/978-3-031-68208-7_20)

### 3.2 PatchTST Architecture

> "PatchTST vectorizes time series into patches of a given size and encodes the resulting sequence of vectors via a Transformer. It segments time series into subseries-level patches served as input tokens." - [Hugging Face](https://huggingface.co/docs/transformers/en/model_doc/patchtst)

**Key benefits:**
1. Local semantic information retained in embeddings
2. Quadratically reduced computation and memory
3. Can attend to longer history

**Patching parameters:**
- Patch length: 16-64 (default 64 in original paper)
- Stride: Equal to patch length (non-overlapping) or half (overlapping)

### 3.3 Foundation Models (TimesFM, Lag-Llama)

**Zero-shot capability:** Pre-trained on billions of time points, can forecast without task-specific training.

> "TimesFM contains a 200M parameter transformer-based model trained on over 100 billion real-world time points. It generates accurate forecasts on unseen datasets without additional training." - [Analytics Vidhya](https://www.analyticsvidhya.com/blog/2024/09/timesfm/)

> "Lag-Llama is a general-purpose foundation model for univariate probabilistic time series forecasting based on a decoder-only transformer architecture that uses lags as covariates." - [arXiv](https://arxiv.org/abs/2310.08278)

**Zero-shot vs Fine-tuned:**
- **Zero-shot:** Useful for cold-start, quick experimentation
- **Fine-tuned:** Significantly better for domain-specific patterns

### 3.4 Feature Preprocessing for Transformers

| Model | Scaling | Features | Notes |
|-------|---------|----------|-------|
| **PatchTST** | Instance normalization (RevIN) | Raw or minimal engineered | Channel-independent |
| **TimesNet** | Standard normalization | Raw OHLCV works well | Learns temporal patterns |
| **TimesFM** | Built-in normalization | Univariate only | Use one feature at a time |
| **Lag-Llama** | Built-in | Univariate + lags | Probabilistic outputs |

### 3.5 Specific Recommendations for This Pipeline

**Option A: Use raw OHLCV with patching (simpler)**
```python
# For PatchTST
from transformers import PatchTSTForPrediction

# Use raw OHLCV: Open, High, Low, Close, Volume
# Let the model learn representations
features = ['open', 'high', 'low', 'close', 'volume']

# Patch configuration
patch_length = 32  # ~2.5 hours of 5-min bars
context_length = 512  # ~42 hours lookback
prediction_length = 20  # Match longest label horizon
```

**Option B: Minimal engineered features (recommended for financial)**
```python
# Combine raw with key engineered features
features = [
    # Raw price action
    'open', 'high', 'low', 'close', 'volume',
    # Key indicators that capture different aspects
    'return_10', 'rsi_14', 'atr_14', 'obv_ratio',
    # Regime indicators
    'volatility_regime', 'trend_regime'
]
```

**Option C: Foundation model for baseline**
```python
# Use TimesFM for quick baseline
from timesfm import TimesFM

model = TimesFM(horizon=20)
# Only needs close price (univariate)
predictions = model.forecast(close_prices)
```

**Sources:**
- [PatchTST Paper](https://arxiv.org/abs/2211.14730)
- [Hugging Face PatchTST](https://huggingface.co/docs/transformers/en/model_doc/patchtst)
- [TimesFM Overview](https://www.analyticsvidhya.com/blog/2024/09/timesfm/)
- [Lag-Llama GitHub](https://github.com/time-series-foundation-models/lag-llama)
- [Raw OHLCV Study](https://arxiv.org/abs/2504.02249)

---

## 4. Classical ML (Random Forest, SVM, Logistic Regression)

### 4.1 Random Forest

**Scaling:** Not required (tree-based)

**Optimal feature count:** 20-80 after selection

> "RF models have some drawbacks, such as they do not model temporal dependencies well. To overcome these limitations, researchers have suggested blended forecasting using RF in combination with time series methodologies." - [PLOS One](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0323015)

**Feature selection:**
- Built-in importance (Gini, permutation)
- Works well with Boruta wrapper
- Random Forest Subset (RFS) selection

> "RFS feature selection efficiently reduces the dimensions of the data by selecting only the best subset of features, which eventually prevents the overfitting problem." - [PLOS One](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0323015)

### 4.2 Support Vector Machine (SVM)

**Scaling:** REQUIRED (StandardScaler preferred)

> "In SVM, feature scaling or normalization are not strictly required, but are highly recommended, as it can significantly improve model performance and convergence speed." - [Forecastegy](https://forecastegy.com/posts/does-svm-need-feature-scaling-or-normalization/)

> "Training an SVM over the scaled and non-scaled data leads to the generation of different models. The new SVM model trained with standardized data has a much higher accuracy of 98%." - [Baeldung](https://www.baeldung.com/cs/svm-feature-scaling)

**Optimal feature count:** 10-50 (SVMs struggle with high dimensionality)

**Best features:**
- Normalized/bounded indicators
- Low correlation between features
- Stationary features

### 4.3 Logistic Regression

**Scaling:** Required (StandardScaler)

**Optimal feature count:** 10-30 (fewer is better for interpretability)

**Feature requirements:**
- Linear relationships with log-odds
- Stationary, normalized
- L1 regularization for automatic feature selection

### 4.4 Specific Recommendations for This Pipeline

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE

# Random Forest - no scaling needed
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)

# SVM - scaling required
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

svm = SVC(kernel='rbf', C=1.0)
svm.fit(X_train_scaled, y_train)

# Logistic with L1 for feature selection
lr = LogisticRegression(penalty='l1', solver='saga', C=0.1)
lr.fit(X_train_scaled, y_train)

# Feature count recommendation
# For SVM/Logistic: use RFE to select top 30 features
rfe = RFE(estimator=svm, n_features_to_select=30)
X_selected = rfe.fit_transform(X_train_scaled, y_train)
```

**Sources:**
- [Random Forest Feature Scaling](https://forecastegy.com/posts/does-random-forest-need-feature-scaling-or-normalization/)
- [SVM Feature Scaling](https://forecastegy.com/posts/does-svm-need-feature-scaling-or-normalization/)
- [Two-Stage RF Forecasting](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0323015)
- [scikit-learn SVM](https://scikit-learn.org/stable/modules/svm.html)

---

## 5. Ensemble Meta-Learners (Stacking)

### 5.1 Meta-Learner Input Features

**Primary input:** Out-of-fold predictions from base models

> "The meta-model is trained using the predictions from the base models as new features. The target output stays the same and the meta-model learns how to combine the base model predictions." - [GeeksforGeeks](https://www.geeksforgeeks.org/machine-learning/stacking-in-machine-learning/)

> "For each instance, the base models produced predictions, which were then used as input features for the meta-learner." - [arXiv Financial Fraud](https://arxiv.org/html/2505.10050v1)

### 5.2 Should Original Features Be Included?

**Generally NO for this pipeline:**

> "Incorporating original features into the second-layer model did not significantly enhance performance, likely because the first-layer model had already extracted most of the critical features." - [Nature](https://www.nature.com/articles/s41529-024-00508-z)

**Exception:** Add a few orthogonal features the base models might miss:
- Regime indicators (if not used by all base models)
- Calendar features
- Macro/external features

### 5.3 Base Model Diversity

> "The biggest gains are usually produced when stacking base learners that have high variability, and uncorrelated, predicted values. The more similar the predicted values are between the base learners, the less advantage there is to combining them." - [HOML](https://bradleyboehmke.github.io/HOML/stacking.html)

**Recommended base model combinations for this pipeline:**
1. XGBoost (gradient boosting)
2. LSTM (sequential patterns)
3. Random Forest (ensemble of trees)
4. Logistic Regression (linear baseline)

### 5.4 Meta-Learner Choice

| Meta-Learner | Pros | Cons |
|--------------|------|------|
| **Logistic Regression** | Simple, interpretable | Linear only |
| **XGBoost** | Handles non-linear | May overfit with few base models |
| **Neural Network** | Flexible | Requires more data |
| **Simple Average** | No training needed | No learning of optimal weights |

> "A stacking model using k-nearest neighbors as the meta-learner alongside seven base learners delivered coefficient of determination of 0.959." - [Nature](https://www.nature.com/articles/s41529-024-00508-z)

### 5.5 Critical: Cross-Validation for Predictions

**Never train meta-learner on same predictions used to train base models.**

```python
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import StackingClassifier

# Option 1: Use sklearn's StackingClassifier (handles CV internally)
stacking = StackingClassifier(
    estimators=[
        ('xgb', XGBClassifier()),
        ('rf', RandomForestClassifier()),
        ('lr', LogisticRegression())
    ],
    final_estimator=LogisticRegression(),
    cv=5  # Critical: uses 5-fold CV for base predictions
)

# Option 2: Manual stacking with time-series CV
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
meta_features = np.zeros((len(X_train), n_base_models))

for train_idx, val_idx in tscv.split(X_train):
    for i, model in enumerate(base_models):
        model.fit(X_train[train_idx], y_train[train_idx])
        meta_features[val_idx, i] = model.predict_proba(X_train[val_idx])[:, 1]

meta_model.fit(meta_features, y_train)
```

### 5.6 Specific Recommendations for This Pipeline

```python
# Recommended stacking architecture
base_models = {
    'xgboost': XGBClassifier(n_estimators=100),
    'lstm': LSTMClassifier(hidden_size=64, num_layers=2),
    'rf': RandomForestClassifier(n_estimators=100),
    'logistic': LogisticRegression(C=0.1)
}

# Meta-learner features
meta_features = [
    # Base model predictions (probabilities)
    'xgb_prob', 'lstm_prob', 'rf_prob', 'logistic_prob',
    # Optional: regime indicators only
    'volatility_regime', 'trend_regime'
]

# Simple but effective meta-learner
meta_model = LogisticRegression(C=1.0)
```

**Sources:**
- [Stacking Ensemble ML](https://machinelearningmastery.com/stacking-ensemble-machine-learning-with-python/)
- [Stacking in ML](https://www.geeksforgeeks.org/machine-learning/stacking-in-machine-learning/)
- [Financial Fraud Detection with Stacking](https://arxiv.org/html/2505.10050v1)
- [HOML Stacking Chapter](https://bradleyboehmke.github.io/HOML/stacking.html)

---

## 6. Pipeline-Specific Recommendations

### 6.1 Feature Groups by Model Family

Based on the ~150-200 features in this pipeline:

```python
# Group 1: For all models (core predictive features)
CORE_FEATURES = [
    # Returns
    'return_1', 'return_5', 'return_10', 'return_20',
    # Momentum (bounded)
    'rsi_14', 'rsi_28', 'macd_hist', 'stoch_k', 'stoch_d',
    'williams_r', 'roc_10', 'cci_20', 'mfi_14',
    # Volatility
    'atr_14', 'atr_ratio', 'bb_position', 'bb_width',
    'parkinson_vol', 'garman_klass_vol',
    # Volume
    'volume_ratio', 'obv_slope', 'vwap_distance',
    # Trend
    'adx_14', 'trend_strength',
]

# Group 2: Additional for tree-based (can handle more features)
TREE_ADDITIONAL = [
    # All MTF features
    'mtf_15m_*', 'mtf_30m_*', 'mtf_1h_*',
    # Wavelets
    'wavelet_*',
    # Microstructure
    'spread_*', 'liquidity_*', 'price_impact_*',
    # Regime
    'volatility_regime', 'trend_regime',
]

# Group 3: For neural networks (normalized, sequential)
NEURAL_FEATURES = CORE_FEATURES + [
    # Wavelets (excellent for LSTMs)
    'wavelet_d1_energy', 'wavelet_d2_energy',
    'wavelet_d3_energy', 'wavelet_approx_trend',
    # Temporal encoding
    'hour_sin', 'hour_cos', 'minute_sin', 'minute_cos',
    'session_asian', 'session_european', 'session_us',
]

# Group 4: For transformers (minimal or raw)
TRANSFORMER_MINIMAL = [
    'open', 'high', 'low', 'close', 'volume',
    'return_10', 'rsi_14', 'atr_14', 'volatility_regime'
]

# Group 5: For SVM/Logistic (small, normalized set)
LINEAR_FEATURES = [
    'return_5', 'return_10',
    'rsi_14', 'macd_hist_norm', 'stoch_k',
    'atr_ratio', 'bb_position',
    'volume_ratio', 'adx_14',
    'hour_sin', 'hour_cos'
]
```

### 6.2 Scaling Strategy by Model

```python
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler

def get_scaler_for_model(model_type: str):
    """Return appropriate scaler for model type."""
    if model_type in ['xgboost', 'lightgbm', 'catboost', 'random_forest']:
        return None  # No scaling needed
    elif model_type in ['lstm', 'gru', 'temporal_cnn']:
        return RobustScaler()  # Handles outliers in financial data
    elif model_type in ['svm', 'logistic']:
        return StandardScaler()  # Standard choice
    elif model_type in ['patchtst', 'timesnet']:
        return None  # Built-in normalization
    else:
        return StandardScaler()  # Safe default
```

### 6.3 Feature Selection Strategy

```python
def select_features_for_model(model_type: str, X: pd.DataFrame, y: pd.Series):
    """Select appropriate features for model type."""

    if model_type in ['xgboost', 'lightgbm', 'catboost']:
        # Use built-in importance, select top 50-100
        model = XGBClassifier(n_estimators=50)
        model.fit(X, y)
        importance = pd.Series(model.feature_importances_, index=X.columns)
        return importance.nlargest(80).index.tolist()

    elif model_type in ['lstm', 'gru']:
        # Pre-defined feature set for sequences
        return [c for c in NEURAL_FEATURES if c in X.columns]

    elif model_type in ['patchtst', 'timesnet']:
        # Minimal features or raw OHLCV
        return [c for c in TRANSFORMER_MINIMAL if c in X.columns]

    elif model_type in ['svm', 'logistic']:
        # Small, normalized set
        return [c for c in LINEAR_FEATURES if c in X.columns]

    elif model_type == 'random_forest':
        # Similar to boosting but fewer features
        model = RandomForestClassifier(n_estimators=50)
        model.fit(X, y)
        importance = pd.Series(model.feature_importances_, index=X.columns)
        return importance.nlargest(50).index.tolist()

    else:
        return X.columns.tolist()
```

### 6.4 Summary Decision Matrix

| Decision | Boosting | Neural | Transformer | Classical | Ensemble |
|----------|----------|--------|-------------|-----------|----------|
| Scale features? | No | Yes (Robust) | No (built-in) | Yes (Standard) | Depends |
| Use wavelets? | Yes | Yes (excellent) | Optional | Maybe | Via base |
| Use MTF features? | Yes | Yes | Optional | Maybe | Via base |
| Use microstructure? | Yes | Yes | Optional | Maybe | Via base |
| Use raw OHLCV? | As ratios | Scaled | Yes | No | Via base |
| Feature count | 50-100 | 20-50 | 5-15 | 10-30 | 4-8 |
| Handle categoricals | Encode/native | Encode | Embed | Encode | Via base |

---

## 7. Research Sources

### Boosting Models
- [Forecasting with Skforecast, XGBoost, LightGBM, CatBoost](https://cienciadedatos.net/documentos/py39-forecasting-time-series-with-skforecast-xgboost-lightgbm-catboost.html)
- [XGBoost vs CatBoost Real-time Trading](https://emergentmethods.medium.com/real-time-head-to-head-adaptive-modeling-of-financial-market-data-using-xgboost-and-catboost-995a115a7495)
- [CatBoost Categorical Features](https://catboost.ai/docs/en/features/categorical-features)
- [LightGBM Feature Importance](https://mljourney.com/lightgbm-feature-importance-comprehensive-guide/)
- [XGBoost Feature Selection](https://xgboosting.com/use-xgboost-feature-importance-for-feature-selection/)

### Neural Networks
- [LSTM Data Preprocessing](https://neuralbrainworks.com/lstm-data-preprocessing-techniques-for-time-series-models/)
- [Scaling for LSTM](https://machinelearningmastery.com/how-to-scale-data-for-long-short-term-memory-networks-in-python/)
- [Novel Wavelet-LSTM Approach](https://link.springer.com/article/10.1007/s00521-024-10561-z)
- [WaveFLSTM Model](https://link.springer.com/article/10.1007/s00521-024-10622-3)
- [Building RNN/LSTM/GRU PyTorch](https://towardsdatascience.com/building-rnn-lstm-and-gru-for-time-series-using-pytorch-a46e5b094e7b/)

### Transformers
- [PatchTST Paper](https://arxiv.org/abs/2211.14730)
- [PatchTST Hugging Face](https://huggingface.co/docs/transformers/en/model_doc/patchtst)
- [TimesFM Overview](https://www.analyticsvidhya.com/blog/2024/09/timesfm/)
- [Lag-Llama](https://arxiv.org/abs/2310.08278)
- [Raw OHLCV vs Features Study](https://arxiv.org/abs/2504.02249)
- [Google Research: Few-Shot Time Series](https://research.google/blog/time-series-foundation-models-can-be-few-shot-learners/)

### Classical ML
- [Random Forest Scaling](https://forecastegy.com/posts/does-random-forest-need-feature-scaling-or-normalization/)
- [SVM Feature Scaling](https://forecastegy.com/posts/does-svm-need-feature-scaling-or-normalization/)
- [SVM Scaling Importance](https://www.baeldung.com/cs/svm-feature-scaling)
- [Two-Stage RF Forecasting](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0323015)

### Ensemble
- [Stacking Ensemble ML](https://machinelearningmastery.com/stacking-ensemble-machine-learning-with-python/)
- [Stacking in ML](https://www.geeksforgeeks.org/machine-learning/stacking-in-machine-learning/)
- [HOML Stacking](https://bradleyboehmke.github.io/HOML/stacking.html)
- [Financial Fraud Stacking](https://arxiv.org/html/2505.10050v1)

### General Feature Engineering
- [Impact of Feature Scaling](https://arxiv.org/html/2506.08274v1)
- [Feature Engineering for Time Series](https://medium.com/data-science-at-microsoft/introduction-to-feature-engineering-for-time-series-forecasting-620aa55fcab0)
- [scikit-learn Scaling Importance](https://scikit-learn.org/stable/auto_examples/preprocessing/plot_scaling_importance.html)

---

*Document generated: 2024-12-24*
*Pipeline version: Phase 1 Complete*
