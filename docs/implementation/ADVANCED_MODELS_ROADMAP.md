# Advanced Models Implementation Roadmap

**Purpose:** Detailed implementation plan for 6 advanced models (13 → 19 model expansion)
**Timeline:** 6-8 weeks (1 engineer) | 4-5 weeks (2 engineers)
**Status:** Planning phase
**Target Completion:** Q1 2025

---

## Table of Contents

1. [Overview](#overview)
2. [Model 14: InceptionTime](#model-14-inceptiontime)
3. [Model 15: 1D ResNet](#model-15-1d-resnet)
4. [Model 16: PatchTST](#model-16-patchtst)
5. [Model 17: iTransformer](#model-17-itransformer)
6. [Model 18: TFT (Temporal Fusion Transformer)](#model-18-tft-temporal-fusion-transformer)
7. [Model 19: N-BEATS](#model-19-n-beats)
8. [Implementation Order & Dependencies](#implementation-order--dependencies)
9. [Testing Strategy](#testing-strategy)
10. [Integration Checklist](#integration-checklist)

---

## Overview

### Goal: Expand from 13 → 19 Models

**Current (13 models):**
- Boosting: XGBoost, LightGBM, CatBoost (3)
- Neural: LSTM, GRU, TCN, Transformer (4)
- Classical: Random Forest, Logistic, SVM (3)
- Ensemble: Voting, Stacking, Blending (3)

**Adding (6 models):**
- CNN: InceptionTime, 1D ResNet (2)
- Advanced Transformers: PatchTST, iTransformer, TFT (3)
- MLP: N-BEATS (1)

**Why these 6?**
- InceptionTime: Multi-scale pattern detection (SOTA for time series classification)
- 1D ResNet: Deep residual learning (proven stable baseline)
- PatchTST: SOTA for long-term forecasting (21% MSE reduction)
- iTransformer: Multivariate correlations (features as tokens)
- TFT: Interpretable attention + quantile forecasting
- N-BEATS: Interpretable decomposition + M4 competition winner

**Total effort:** 14-18 days (112-144 hours)

---

## Model 14: InceptionTime

### Overview

**Use Case:** Multi-scale pattern detection via parallel convolutional kernels
**Architecture:** Ensemble of Inception modules (5 kernel sizes: 10, 20, 40)
**Input:** 3D tensor `(n_samples, seq_len, n_features)`
**Output:** Class probabilities + confidence
**Effort:** 3 days (24 hours)

### Why InceptionTime?

- **Multi-scale learning:** Captures patterns at different temporal scales simultaneously
- **Proven SOTA:** Won UCR time series classification benchmark (85% datasets)
- **Robust:** Ensemble-in-architecture reduces overfitting
- **Fast training:** Parallelizable convolutions, GPU-efficient

### Architecture Details

```python
# src/models/cnn/inceptiontime.py

class InceptionModule(nn.Module):
    """
    Single Inception module with parallel convolutions.

    Kernel sizes: 10, 20, 40 (for 5min bars)
    Each kernel captures different pattern scales:
    - 10: Short-term (50 minutes)
    - 20: Medium-term (100 minutes)
    - 40: Long-term (200 minutes)
    """
    def __init__(self, in_channels, out_channels=32, bottleneck=32):
        super().__init__()

        # Bottleneck layer (reduce channels for efficiency)
        self.bottleneck = nn.Conv1d(in_channels, bottleneck, kernel_size=1, bias=False)

        # Parallel convolutions with different kernel sizes
        self.conv_10 = nn.Conv1d(bottleneck, out_channels, kernel_size=10, padding=5, bias=False)
        self.conv_20 = nn.Conv1d(bottleneck, out_channels, kernel_size=20, padding=10, bias=False)
        self.conv_40 = nn.Conv1d(bottleneck, out_channels, kernel_size=40, padding=20, bias=False)

        # Max pooling branch
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.conv_pool = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)

        # Batch norm + ReLU
        self.bn = nn.BatchNorm1d(out_channels * 4)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Bottleneck
        x_bottleneck = self.bottleneck(x)

        # Parallel branches
        conv_10_out = self.conv_10(x_bottleneck)
        conv_20_out = self.conv_20(x_bottleneck)
        conv_40_out = self.conv_40(x_bottleneck)

        # Pooling branch
        x_pool = self.maxpool(x)
        conv_pool_out = self.conv_pool(x_pool)

        # Concatenate all branches
        out = torch.cat([conv_10_out, conv_20_out, conv_40_out, conv_pool_out], dim=1)

        # Batch norm + activation
        out = self.bn(out)
        out = self.relu(out)

        return out

class InceptionTimeModel(BaseModel):
    """
    InceptionTime: Ensemble of 5 Inception networks.

    Each network has 6 Inception modules stacked.
    Final prediction is average of 5 networks.
    """
    def __init__(self, config=None):
        super().__init__(config)

        # Build 5 independent networks (ensemble)
        self.num_networks = 5
        self.networks = nn.ModuleList([
            self._build_single_network() for _ in range(self.num_networks)
        ])

    def _build_single_network(self):
        """Build one Inception network (6 modules)."""
        n_features = self.config.get('n_features', 25)
        num_filters = self.config.get('num_filters', 32)
        num_modules = self.config.get('num_modules', 6)

        modules = []
        in_channels = n_features

        for i in range(num_modules):
            modules.append(InceptionModule(
                in_channels=in_channels if i == 0 else num_filters * 4,
                out_channels=num_filters,
                bottleneck=32
            ))

        # Global average pooling + classifier
        modules.append(nn.AdaptiveAvgPool1d(1))
        modules.append(nn.Flatten())
        modules.append(nn.Linear(num_filters * 4, 3))  # 3 classes

        return nn.Sequential(*modules)

    def fit(self, X_train, y_train, X_val, y_val, sample_weights=None, config=None):
        """Train ensemble of 5 networks."""
        self._validate_input_shape(X_train, expected_ndim=3)

        # Convert to PyTorch tensors
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.LongTensor(y_train).to(self.device)
        X_val_t = torch.FloatTensor(X_val).to(self.device)
        y_val_t = torch.LongTensor(y_val).to(self.device)

        # Permute to (batch, channels, seq_len) for Conv1d
        X_train_t = X_train_t.permute(0, 2, 1)
        X_val_t = X_val_t.permute(0, 2, 1)

        # Train each network independently
        for net_idx, network in enumerate(self.networks):
            logger.info(f"Training network {net_idx + 1}/{self.num_networks}")

            optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
            criterion = nn.CrossEntropyLoss()

            best_val_loss = float('inf')
            patience_counter = 0

            for epoch in range(100):
                # Train
                network.train()
                train_loss = self._train_epoch(network, X_train_t, y_train_t, optimizer, criterion)

                # Validate
                network.eval()
                with torch.no_grad():
                    val_logits = network(X_val_t)
                    val_loss = criterion(val_logits, y_val_t).item()

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= 10:
                        break

        return TrainingMetrics(
            train_loss=train_loss,
            val_loss=best_val_loss,
            best_iteration=epoch,
            training_time=time.time() - start_time
        )

    def predict(self, X):
        """Ensemble prediction (average of 5 networks)."""
        self._validate_input_shape(X, expected_ndim=3)

        X_t = torch.FloatTensor(X).to(self.device).permute(0, 2, 1)

        # Get predictions from all networks
        all_probs = []
        for network in self.networks:
            network.eval()
            with torch.no_grad():
                logits = network(X_t)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                all_probs.append(probs)

        # Average ensemble
        avg_probs = np.mean(all_probs, axis=0)
        predictions = np.argmax(avg_probs, axis=1)
        confidence = np.max(avg_probs, axis=1)

        return PredictionOutput(predictions, avg_probs, confidence)
```

### Implementation Steps

**Day 1: Core Architecture**
1. Create `src/models/cnn/` directory
2. Implement `InceptionModule` class
3. Implement `InceptionTimeModel` (BaseModel interface)
4. Add to model registry

**Day 2: Training & Integration**
5. Implement fit() with ensemble training
6. Implement predict() with ensemble averaging
7. Add save/load methods
8. Create config YAML (`config/models/inceptiontime.yaml`)

**Day 3: Testing & Validation**
9. Unit tests (fit, predict, save/load)
10. Integration test with Phase 1 data
11. Hyperparameter search space (`param_spaces.py`)
12. Documentation updates

### Config Example

```yaml
# config/models/inceptiontime.yaml
model_name: inceptiontime
family: cnn

# Architecture
num_networks: 5        # Ensemble size
num_modules: 6         # Inception modules per network
num_filters: 32        # Filters per convolution
bottleneck: 32         # Bottleneck channels

# Training
learning_rate: 0.001
batch_size: 64
max_epochs: 100
patience: 10

# Input
seq_len: 60
n_features: 25

# MTF Strategy
mtf_strategy: single_tf  # Or mtf_ingestion
training_timeframe: 15min
```

### Expected Performance

- Training time: 30-60 min (GPU)
- Inference: 5-10ms per sample (GPU)
- Memory: ~4GB GPU for typical configs
- Sharpe improvement: +0.1 to +0.3 vs LSTM

---

## Model 15: 1D ResNet

### Overview

**Use Case:** Deep residual learning for temporal patterns
**Architecture:** 1D ResNet with skip connections (prevents vanishing gradients)
**Input:** 3D tensor `(n_samples, seq_len, n_features)`
**Output:** Class probabilities
**Effort:** 2 days (16 hours)

### Why 1D ResNet?

- **Deep networks:** Skip connections enable 20+ layers
- **Proven stable:** ResNet architecture widely validated
- **Fast convergence:** Residual learning speeds up training
- **Robust baseline:** Good comparison point for other models

### Architecture Details

```python
# src/models/cnn/resnet1d.py

class ResidualBlock(nn.Module):
    """
    Residual block with skip connection.

    Two 1D convolutions + batch norm + ReLU + skip connection
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=kernel_size//2, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               stride=1, padding=kernel_size//2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # Skip connection (project if channels change)
        if in_channels != out_channels or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        residual = self.skip(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual  # Skip connection
        out = self.relu(out)

        return out

class ResNet1D(BaseModel):
    """
    1D ResNet for time series classification.

    Architecture: [64, 128, 256, 512] filters across 4 blocks
    Each block has 2-3 residual units
    """
    def __init__(self, config=None):
        super().__init__(config)

        n_features = self.config.get('n_features', 25)
        num_blocks = self.config.get('num_blocks', 4)
        base_filters = self.config.get('base_filters', 64)

        # Initial conv
        self.conv1 = nn.Conv1d(n_features, base_filters, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(base_filters)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # Residual blocks
        self.layer1 = self._make_layer(base_filters, base_filters, num_blocks=2)
        self.layer2 = self._make_layer(base_filters, base_filters*2, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(base_filters*2, base_filters*4, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(base_filters*4, base_filters*8, num_blocks=2, stride=2)

        # Global average pooling + classifier
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(base_filters*8, 3)  # 3 classes

    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        """Create a layer of residual blocks."""
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride=stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        # x: (batch, seq_len, features) → (batch, features, seq_len)
        x = x.permute(0, 2, 1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
```

### Implementation Steps

**Day 1: Architecture**
1. Implement `ResidualBlock`
2. Implement `ResNet1D` (BaseModel interface)
3. Add to registry
4. Create config YAML

**Day 2: Testing**
5. Implement fit/predict/save/load
6. Unit tests
7. Integration tests
8. Hyperparameter space

### Expected Performance

- Training time: 20-40 min (GPU)
- Inference: 5-10ms (GPU)
- Memory: ~3GB GPU
- Sharpe: Comparable to TCN

---

## Model 16: PatchTST

### Overview

**Use Case:** SOTA long-term forecasting via patch-based attention
**Architecture:** Divide sequence into patches → Transformer on patches
**Input:** 3D tensor `(n_samples, seq_len, n_features)`
**Output:** Probabilistic forecasts with quantiles
**Effort:** 4 days (32 hours)

### Why PatchTST?

- **SOTA performance:** 21% MSE reduction vs vanilla Transformer on long-term forecasting
- **Efficient attention:** O(L/P)² instead of O(L)² where P = patch_len
- **Multi-scale:** Patches capture both local and global patterns
- **Production-safe:** Causal masks prevent lookahead

### Architecture Details

```python
# src/models/transformers/patchtst.py

class PatchEmbedding(nn.Module):
    """
    Divide sequence into patches and embed.

    Example: seq_len=512, patch_len=16, stride=8
    → 63 patches (with overlap)
    """
    def __init__(self, patch_len=16, stride=8, d_model=128, n_features=5):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride

        # Linear projection per feature
        self.proj = nn.Linear(patch_len * n_features, d_model)

    def forward(self, x):
        # x: (batch, seq_len, n_features)
        batch_size, seq_len, n_features = x.shape

        # Unfold into patches
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        # patches: (batch, num_patches, n_features, patch_len)

        # Flatten patch
        patches = patches.reshape(batch_size, -1, self.patch_len * n_features)

        # Project to d_model
        embedded = self.proj(patches)
        # embedded: (batch, num_patches, d_model)

        return embedded

class PatchTST(BaseModel):
    """
    PatchTST: Patch-based Transformer for time series.

    Key innovation: Patches reduce sequence length from L → L/P,
    making attention O((L/P)²) instead of O(L²).
    """
    def __init__(self, config=None):
        super().__init__(config)

        # Config
        patch_len = self.config.get('patch_len', 16)
        stride = self.config.get('stride', 8)
        d_model = self.config.get('d_model', 128)
        nhead = self.config.get('nhead', 8)
        num_layers = self.config.get('num_layers', 3)
        n_features = self.config.get('n_features', 5)

        # Patch embedding
        self.patch_embed = PatchEmbedding(patch_len, stride, d_model, n_features)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model*4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, 3)  # 3 classes
        )

    def forward(self, x):
        # x: (batch, seq_len, n_features)

        # Patch embedding
        x = self.patch_embed(x)  # (batch, num_patches, d_model)

        # Positional encoding
        x = self.pos_encoder(x)

        # Transformer
        x = self.transformer(x)  # (batch, num_patches, d_model)

        # Global average pooling
        x = x.mean(dim=1)  # (batch, d_model)

        # Classifier
        logits = self.fc(x)  # (batch, 3)

        return logits
```

### Implementation Steps

**Day 1-2: Core**
1. Implement `PatchEmbedding`
2. Implement `PatchTST` model
3. Add causal masking (prevent lookahead)
4. Registry integration

**Day 3: Training**
5. Fit/predict methods
6. Save/load
7. Config YAML

**Day 4: Testing**
8. Unit tests
9. Integration tests
10. Hyperparameter space
11. Docs

### Expected Performance

- Training: 1-2 hours (GPU)
- Inference: 10-30ms (GPU)
- Memory: ~8GB GPU (seq_len=512)
- Sharpe: +0.2 to +0.4 vs vanilla Transformer

---

## Model 17: iTransformer

### Overview

**Use Case:** Multivariate correlations via inverted attention
**Architecture:** Features as tokens (not timesteps)
**Input:** 3D tensor `(n_samples, seq_len, n_features)`
**Output:** Class probabilities
**Effort:** 3 days (24 hours)

### Why iTransformer?

- **Novel approach:** Attention over features, not time
- **Multivariate correlations:** Learns feature interactions across time
- **Efficient:** Scales with O(F²) not O(L²) where F << L
- **Complementary:** Different inductive bias than PatchTST

### Architecture Details

```python
# src/models/transformers/itransformer.py

class iTransformer(BaseModel):
    """
    iTransformer: Inverted Transformer.

    Standard Transformer: tokens = timesteps, attention across time
    iTransformer: tokens = features, attention across features

    This allows learning multivariate correlations.
    """
    def __init__(self, config=None):
        super().__init__(config)

        d_model = self.config.get('d_model', 128)
        nhead = self.config.get('nhead', 8)
        num_layers = self.config.get('num_layers', 4)
        seq_len = self.config.get('seq_len', 60)

        # Feature embedding (project each feature's time series to d_model)
        self.feature_embed = nn.Linear(seq_len, d_model)

        # Transformer encoder (attention over features)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model*4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classifier
        self.fc = nn.Linear(d_model, 3)  # 3 classes

    def forward(self, x):
        # x: (batch, seq_len, n_features)

        # Transpose: (batch, n_features, seq_len)
        # Now each "token" is one feature's full time series
        x = x.transpose(1, 2)

        # Embed each feature
        x = self.feature_embed(x)  # (batch, n_features, d_model)

        # Transformer (attention over features)
        x = self.transformer(x)  # (batch, n_features, d_model)

        # Aggregate across features
        x = x.mean(dim=1)  # (batch, d_model)

        # Classify
        logits = self.fc(x)

        return logits
```

### Implementation Steps

**Day 1: Core**
1. Implement `iTransformer` model
2. Feature embedding layer
3. Registry integration

**Day 2: Training**
4. Fit/predict
5. Save/load
6. Config

**Day 3: Testing**
7. Tests
8. Integration
9. Hyperparameter space

---

## Model 18: TFT (Temporal Fusion Transformer)

### Overview

**Use Case:** Interpretable attention + quantile forecasting
**Architecture:** Variable selection + gating + multi-head attention
**Input:** 3D + static/known covariates
**Output:** Point forecasts + quantiles (q05, q50, q95)
**Effort:** 5 days (40 hours)

### Why TFT?

- **Interpretability:** Variable selection network shows feature importance
- **Quantile forecasting:** Native uncertainty quantification
- **Multi-horizon:** Single model predicts multiple horizons
- **Production-proven:** Google uses for forecasting

### Architecture (Complex - Simplified for MVP)

```python
# src/models/transformers/tft.py

class TFT(BaseModel):
    """
    Temporal Fusion Transformer (simplified implementation).

    Full TFT is complex. We implement core components:
    1. Variable selection network
    2. LSTM encoder
    3. Multi-head attention
    4. Quantile output layers
    """
    def __init__(self, config=None):
        super().__init__(config)

        hidden_size = self.config.get('hidden_size', 128)
        num_attention_heads = self.config.get('num_attention_heads', 4)

        # Variable selection (simplified: just a gating network)
        self.variable_selection = nn.Sequential(
            nn.Linear(n_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_features),
            nn.Sigmoid()  # Weights for each feature
        )

        # LSTM encoder
        self.encoder = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True
        )

        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_attention_heads,
            batch_first=True
        )

        # Quantile output heads (3 quantiles: 0.05, 0.5, 0.95)
        self.fc_q05 = nn.Linear(hidden_size, 3)
        self.fc_q50 = nn.Linear(hidden_size, 3)
        self.fc_q95 = nn.Linear(hidden_size, 3)

    def forward(self, x):
        # Variable selection
        weights = self.variable_selection(x.mean(dim=1))  # (batch, n_features)
        x_selected = x * weights.unsqueeze(1)  # (batch, seq_len, n_features)

        # LSTM encoding
        lstm_out, _ = self.encoder(x_selected)  # (batch, seq_len, hidden)

        # Self-attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)

        # Take last timestep
        last_hidden = attn_out[:, -1, :]  # (batch, hidden)

        # Quantile forecasts
        q05_logits = self.fc_q05(last_hidden)
        q50_logits = self.fc_q50(last_hidden)  # Median = point forecast
        q95_logits = self.fc_q95(last_hidden)

        return q50_logits  # Return median for classification
```

### Implementation Steps

**Day 1-2: Core**
1. Implement variable selection
2. LSTM encoder
3. Attention layer

**Day 3: Quantiles**
4. Quantile output heads
5. Quantile loss function

**Day 4: Training**
6. Fit/predict
7. Save/load
8. Config

**Day 5: Testing**
9. Tests
10. Integration
11. Docs

---

## Model 19: N-BEATS

### Overview

**Use Case:** Interpretable trend/seasonal decomposition
**Architecture:** Stacked blocks with basis expansion
**Input:** 3D tensor
**Output:** Trend + Seasonal components
**Effort:** 1 day (8 hours)

### Why N-BEATS?

- **M4 Competition Winner:** Beat all statistical + ML methods
- **Interpretability:** Explicit trend/seasonal decomposition
- **Fast:** Pure MLP (no RNN/attention overhead)
- **Doubly residual:** Both forward and backward forecasts

### Architecture Details

```python
# src/models/mlp/nbeats.py

class NBEATSBlock(nn.Module):
    """
    Single N-BEATS block.

    Takes input, outputs:
    - Backcast: Reconstruction of input (residual learning)
    - Forecast: Prediction
    """
    def __init__(self, input_size, theta_size, basis_function, hidden_size=256):
        super().__init__()

        # FC stack (4 layers)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)

        # Theta layers (expansion coefficients)
        self.theta_b = nn.Linear(hidden_size, theta_size)  # Backcast
        self.theta_f = nn.Linear(hidden_size, theta_size)  # Forecast

        self.basis = basis_function

    def forward(self, x):
        # FC stack with ReLU
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        h = F.relu(self.fc4(h))

        # Theta coefficients
        theta_b = self.theta_b(h)
        theta_f = self.theta_f(h)

        # Basis expansion
        backcast = self.basis(theta_b, is_forecast=False)
        forecast = self.basis(theta_f, is_forecast=True)

        return backcast, forecast

class NBEATS(BaseModel):
    """
    N-BEATS: Neural Basis Expansion Analysis for Time Series.

    2 stacks:
    - Trend stack (polynomial basis)
    - Seasonality stack (Fourier basis)
    """
    def __init__(self, config=None):
        super().__init__(config)

        # Trend stack (3 blocks with polynomial basis)
        self.trend_blocks = nn.ModuleList([
            NBEATSBlock(input_size, theta_size=4, basis_function=polynomial_basis)
            for _ in range(3)
        ])

        # Seasonal stack (3 blocks with Fourier basis)
        self.seasonal_blocks = nn.ModuleList([
            NBEATSBlock(input_size, theta_size=8, basis_function=fourier_basis)
            for _ in range(3)
        ])

        # Classifier (on combined forecast)
        self.fc = nn.Linear(forecast_horizon, 3)
```

### Implementation Steps

**Day 1:**
1. Implement `NBEATSBlock`
2. Implement basis functions (polynomial, Fourier)
3. Implement `NBEATS` model
4. Fit/predict/save/load
5. Tests
6. Config

---

## Implementation Order & Dependencies

### Recommended Sequence

```
Week 1: CNN Models (Easier, GPU warmup)
├── Day 1-2: 1D ResNet      (Simpler, good baseline)
└── Day 3-5: InceptionTime  (More complex, ensemble)

Week 2: MLP + Advanced Transformer Start
├── Day 1: N-BEATS          (Fast implementation)
├── Day 2-5: PatchTST       (Most valuable, SOTA)

Week 3: Advanced Transformers
├── Day 1-3: iTransformer   (Novel approach)
└── Day 4-5: TFT (start)

Week 4: TFT Completion
├── Day 1-2: TFT (finish)
└── Day 3-5: Integration testing, docs, cleanup
```

### Parallelization (2 Engineers)

**Engineer A:** CNN + MLP
- Week 1: ResNet1D + InceptionTime
- Week 2: N-BEATS + testing

**Engineer B:** Transformers
- Week 1: PatchTST
- Week 2: iTransformer + TFT

**Both:** Week 3-4 integration, testing, docs

---

## Testing Strategy

### Per-Model Tests

```python
# tests/phase_2_tests/test_<model>.py

def test_model_fit():
    """Test training on synthetic data."""
    pass

def test_model_predict():
    """Test prediction output shape and range."""
    pass

def test_model_save_load():
    """Test persistence."""
    pass

def test_model_input_validation():
    """Test input shape validation."""
    pass

def test_model_with_real_data():
    """Test with Phase 1 container."""
    pass
```

### Integration Tests

```python
# tests/integration/test_advanced_models.py

def test_all_19_models_train():
    """Smoke test: all 19 models can train."""
    for model_name in ModelRegistry.list_all():
        # ... train on small dataset
        assert model can fit and predict

def test_ensemble_compatibility():
    """Test same-family ensemble constraints."""
    # CNN ensemble: inceptiontime + resnet1d
    # Transformer ensemble: patchtst + itransformer + tft
    pass
```

---

## Integration Checklist

### Per Model

- [ ] Implement `BaseModel` interface (fit, predict, save, load)
- [ ] Register with `@register(name, family)` decorator
- [ ] Create config YAML in `config/models/`
- [ ] Add hyperparameter space to `param_spaces.py`
- [ ] Unit tests (5+ test functions)
- [ ] Integration test with Phase 1 data
- [ ] Documentation in model file (docstrings)
- [ ] Update `CLAUDE.md` model count
- [ ] Update `PROJECT_CHARTER.md` with status

### After All 6 Models

- [ ] Update `MODEL_INTEGRATION.md` with examples
- [ ] Run full test suite (`pytest tests/`)
- [ ] Benchmark all 19 models on same dataset
- [ ] Create comparison table (Sharpe, time, memory)
- [ ] Update `MTF_IMPLEMENTATION_ROADMAP.md` (Phase 5 complete)
- [ ] Production deployment guide for advanced models

---

## Summary

**6 Advanced Models:**
1. InceptionTime - Multi-scale CNN (3 days)
2. 1D ResNet - Deep residual (2 days)
3. PatchTST - SOTA Transformer (4 days)
4. iTransformer - Inverted attention (3 days)
5. TFT - Interpretable + quantiles (5 days)
6. N-BEATS - Decomposition MLP (1 day)

**Total: 18 days** (3.5 weeks @ 1 engineer) | **10 days** (2 weeks @ 2 engineers)

**After completion:** 19 production-ready models, all tested, documented, and benchmarked.
