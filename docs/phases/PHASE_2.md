# Phase 2 – Training Base Models (N-HiTS, TFT, PatchTST)

## Overview

Phase 2 trains three independent deep learning models on the Phase 1 labeled dataset. Each model specializes in time-series classification and will become an "expert" in the final ensemble. The three models are:

1. **N-HiTS** (Neural Hierarchical Interpolation for Time Series) – MLP-based multi-scale forecaster
2. **TFT** (Temporal Fusion Transformer) – Attention-based multi-horizon model
3. **PatchTST** (Patch Time Series Transformer) – Modern patch-based transformer

**Phase 2 Goal:** Train each model to predict triple-barrier labels (−1, 0, +1) for horizons (1-bar, 5-bar, 20-bar) with maximum accuracy and Sharpe ratio.

**Think of Phase 2 as:** Teaching three different "quants" to read the same market data, each with their own approach.

---

## Objectives

### Primary Goals
- Configure and train N-HiTS for multi-horizon classification
- Configure and train TFT for multi-horizon classification with interpretability
- Configure and train PatchTST for long-range dependency capture
- Generate out-of-sample predictions (probabilities) on validation set for each model
- Log comprehensive metrics: Accuracy, F1, Sharpe, Max Drawdown per model/horizon
- Save trained model weights and configurations for Phase 3 (ensemble stacking)

### Success Criteria
- Each model achieves validation F1 > 0.35 on at least one horizon
- Each model achieves validation Sharpe > 0.3 on at least one horizon
- No severe overfitting (train/val performance gap < 20%)
- Training completes within reasonable time (<24 hours per model on single GPU)
- All three models generate consistent probability outputs ready for stacking

---

## Prerequisites

### Required Inputs from Phase 1
- `data/final/MES_final_labeled.parquet`
- `data/final/MGC_final_labeled.parquet`
- `config/splits/train_indices.npy`
- `config/splits/val_indices.npy`
- `config/splits/test_indices.npy` (held out, not used in Phase 2)
- `config/ga_results/` (GA-optimized barrier parameters for reference)

### Infrastructure Requirements
- **GPU:** 1× RTX 3090/4090/5090 or A100/H100 (16GB+ VRAM minimum)
- **RAM:** 64–128 GB
- **Storage:** 50–100 GB for checkpoints and logs
- **Compute Time:** ~8-24 hours per model (depending on hyperparameter tuning)

### Software Dependencies
```
# Deep Learning
torch >= 2.0
lightning >= 2.0 (PyTorch Lightning for training loops)

# Time Series Libraries
neuralforecast >= 1.6  (for N-HiTS)
pytorch-forecasting >= 1.0  (for TFT)
# PatchTST: use official repo or neuralforecast integration

# Utilities
numpy >= 1.24
pandas >= 2.0
scikit-learn >= 1.3
vectorbt >= 0.26  (for backtest metrics)
optuna >= 3.0  (optional, for hyperparameter tuning)
wandb or tensorboard  (for experiment tracking)
```

---

## Shared Conventions for All Models

### Data Format Standards

#### Input Shape
- **Sequence Length:** `L` bars of historical context
  - Recommended: `L=128` (starting point)
  - Can tune: `L ∈ {64, 128, 256}`
- **Feature Dimension:** `F` features per bar
  - From Phase 1: typically 40–80 features
- **Batch Shape:** `[B, L, F]` where `B` = batch size

#### Label Format
- **Original labels:** `{−1, 0, +1}` from Phase 1
- **Mapped to class indices:** `{0, 1, 2}`
  - −1 (short/stop hit) → class 0
  - 0 (neutral) → class 1
  - +1 (long/target hit) → class 2
- **Per horizon:**
  - `y_1` ∈ {0, 1, 2} (1-bar horizon)
  - `y_5` ∈ {0, 1, 2} (5-bar horizon)
  - `y_20` ∈ {0, 1, 2} (20-bar horizon)

#### Sample Weights
- From Phase 1: `sample_weight_1`, `sample_weight_5`, `sample_weight_20`
- Used in weighted cross-entropy loss
- Emphasizes high-quality signals (A+/A tier)

### Loss Function

**Multi-Horizon Weighted Cross-Entropy:**

```
total_loss = α₁ * CE(logits_1, y_1, weights=w_1)
           + α₅ * CE(logits_5, y_5, weights=w_5)
           + α₂₀ * CE(logits_20, y_20, weights=w_20)
```

Where:
- `CE` = Cross-Entropy Loss
- `α₁, α₅, α₂₀` = horizon importance weights (typically `[1.0, 1.0, 1.0]`)
- Can adjust if one horizon is noisier (e.g., reduce α₁ if 1-bar is too noisy)

### Output Format

**Each model must output:**
```python
{
  'logits_1': Tensor[B, 3],   # 1-bar horizon class logits
  'logits_5': Tensor[B, 3],   # 5-bar horizon class logits
  'logits_20': Tensor[B, 3],  # 20-bar horizon class logits
}
```

**Convert to probabilities:**
```python
probs_1 = softmax(logits_1, dim=-1)   # [B, 3], sums to 1
probs_5 = softmax(logits_5, dim=-1)
probs_20 = softmax(logits_20, dim=-1)
```

These probabilities become inputs for Phase 4 (ensemble meta-learner).

### Evaluation Metrics

**Classification Metrics (per horizon):**
- **Accuracy:** `correct_predictions / total_predictions`
- **F1 Score:**
  - Macro F1 (average of F1 for each class)
  - Per-class F1 (especially for classes 0 and 2: short/long)
- **Confusion Matrix:** visualize prediction patterns
- **Precision/Recall:** per class

**Trading Metrics (per horizon):**
- Use simple rule-based strategy on validation set:
  - `class_pred = argmax(probs)`
  - If `class_pred == 2` (+1): go long next bar
  - If `class_pred == 0` (−1): go short next bar
  - If `class_pred == 1` (0): no position
- Compute with vectorbt:
  - **Sharpe Ratio:** annualized
  - **Max Drawdown:** percentage
  - **Number of Trades**
  - **Win Rate**
  - **Profit Factor**

**Log these metrics every epoch on train and validation sets.**

### Training Loop Structure

**Standard epoch loop:**
```python
for epoch in range(max_epochs):
    model.train()
    for batch in train_dataloader:
        X, y_1, y_5, y_20, w_1, w_5, w_20 = batch
        
        # Forward
        out = model(X)
        logits_1, logits_5, logits_20 = out['logits_1'], out['logits_5'], out['logits_20']
        
        # Compute loss
        loss_1 = weighted_cross_entropy(logits_1, y_1, weights=w_1)
        loss_5 = weighted_cross_entropy(logits_5, y_5, weights=w_5)
        loss_20 = weighted_cross_entropy(logits_20, y_20, weights=w_20)
        total_loss = loss_1 + loss_5 + loss_20
        
        # Backward
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
    
    # Validation
    model.eval()
    with torch.no_grad():
        val_loss, val_metrics = evaluate(model, val_dataloader)
    
    # Early stopping check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_checkpoint(model)
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            break  # Early stop
    
    # Learning rate scheduler step
    scheduler.step(val_loss)
    
    # Log metrics
    log_metrics(epoch, train_loss, val_loss, val_metrics)
```

---

## Model 1: N-HiTS (Neural Hierarchical Interpolation for Time Series)

### Model Overview

**N-HiTS** is an MLP-based architecture with hierarchical stacking that specializes in capturing multi-scale patterns (trend vs short-term fluctuations).

**Key Features:**
- Doubly residual stacking (stacks of blocks)
- Multi-rate input pooling (coarse to fine)
- Fast training (50× faster than Transformer models)
- Effective on long-horizon forecasting

**Use Case:** Strong baseline, captures multi-scale price patterns efficiently.

### Architecture Configuration

#### Core Hyperparameters

```python
nhits_config = {
    # Architecture
    'n_stacks': 3,              # Number of hierarchical stacks
    'n_blocks': [1, 1, 1],      # Blocks per stack
    'n_pool_kernel_size': [2, 2, 1],  # Pooling kernels per stack
    'n_freq_downsample': [4, 2, 1],    # Frequency downsampling
    
    # MLP size
    'mlp_units': [[512, 512], [512, 512], [512, 512]],  # Hidden units per block
    'activation': 'ReLU',
    
    # Input/Output
    'input_size': 128,          # Sequence length L
    'output_size': 20,          # Max forecast horizon (for 20-bar)
    'n_features': 50,           # Number of input features F
    
    # Regularization
    'dropout': 0.2,
    'batch_normalization': True,
    
    # Training
    'learning_rate': 1e-3,
    'batch_size': 64,
    'max_epochs': 100,
    'early_stopping_patience': 10,
}
```

### Model Adaptation for Classification

**Challenge:** N-HiTS is designed for regression (forecasting continuous values).

**Solution:** Add classification head on top of N-HiTS encoder.

**Modified Architecture:**
```
Input [B, L, F]
    ↓
N-HiTS Encoder (hierarchical stacks)
    ↓
Latent Representation h [B, H_latent]
    ↓
Shared MLP [B, H_mid]
    ↓
    ├── Head_1 (Linear 3 classes) → logits_1 [B, 3]
    ├── Head_5 (Linear 3 classes) → logits_5 [B, 3]
    └── Head_20 (Linear 3 classes) → logits_20 [B, 3]
```

**Implementation Sketch:**
```python
class NHiTSClassifier(nn.Module):
    def __init__(self, nhits_config):
        super().__init__()
        # Use NeuralForecast's NHiTS as encoder
        from neuralforecast.models import NHITS
        self.encoder = NHITS(
            h=nhits_config['output_size'],
            input_size=nhits_config['input_size'],
            n_blocks=nhits_config['n_blocks'],
            mlp_units=nhits_config['mlp_units'],
            n_pool_kernel_size=nhits_config['n_pool_kernel_size'],
            n_freq_downsample=nhits_config['n_freq_downsample'],
            dropout_prob=nhits_config['dropout'],
        )
        
        # Classification heads
        H_latent = 512  # Encoder output dim (from last MLP)
        H_mid = 256
        self.shared_mlp = nn.Sequential(
            nn.Linear(H_latent, H_mid),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.head_1 = nn.Linear(H_mid, 3)
        self.head_5 = nn.Linear(H_mid, 3)
        self.head_20 = nn.Linear(H_mid, 3)
    
    def forward(self, x):
        # x: [B, L, F]
        # Encoder forward (may need adaptation for NeuralForecast API)
        h = self.encoder(x)  # Output latent representation
        
        # Shared representation
        h_shared = self.shared_mlp(h)  # [B, H_mid]
        
        # Multi-horizon heads
        logits_1 = self.head_1(h_shared)
        logits_5 = self.head_5(h_shared)
        logits_20 = self.head_20(h_shared)
        
        return {
            'logits_1': logits_1,
            'logits_5': logits_5,
            'logits_20': logits_20,
        }
```

### Data Preparation for N-HiTS

#### Dataset Class
```python
class NHiTSDataset(torch.utils.data.Dataset):
    def __init__(self, df, indices, seq_len=128, feature_cols=None):
        self.df = df.iloc[indices].reset_index(drop=True)
        self.seq_len = seq_len
        self.feature_cols = feature_cols or [c for c in df.columns if c.startswith('feature_')]
        
        # Filter valid samples (must have seq_len history)
        self.valid_indices = [i for i in range(seq_len, len(self.df))]
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]
        
        # Get sequence window [actual_idx - seq_len : actual_idx]
        seq = self.df.iloc[actual_idx - self.seq_len : actual_idx][self.feature_cols].values  # [L, F]
        
        # Get labels and weights at actual_idx
        y_1 = self.df.iloc[actual_idx]['label_1']  # Map -1,0,+1 → 0,1,2
        y_5 = self.df.iloc[actual_idx]['label_5']
        y_20 = self.df.iloc[actual_idx]['label_20']
        
        w_1 = self.df.iloc[actual_idx]['sample_weight_1']
        w_5 = self.df.iloc[actual_idx]['sample_weight_5']
        w_20 = self.df.iloc[actual_idx]['sample_weight_20']
        
        # Convert labels from {-1,0,+1} to {0,1,2}
        y_1 = y_1 + 1
        y_5 = y_5 + 1
        y_20 = y_20 + 1
        
        return {
            'X': torch.FloatTensor(seq),
            'y_1': torch.LongTensor([y_1])[0],
            'y_5': torch.LongTensor([y_5])[0],
            'y_20': torch.LongTensor([y_20])[0],
            'w_1': torch.FloatTensor([w_1])[0],
            'w_5': torch.FloatTensor([w_5])[0],
            'w_20': torch.FloatTensor([w_20])[0],
        }
```

#### Feature Scaling
- Compute mean/std from **training set only**
- Apply StandardScaler per feature column
- Save scaler for inference:
  ```python
  from sklearn.preprocessing import StandardScaler
  scaler = StandardScaler()
  scaler.fit(train_df[feature_cols])
  joblib.dump(scaler, 'models/nhits_scaler.pkl')
  ```

### Training Procedure

#### Setup
```python
# Load data
train_df = pd.read_parquet('data/final/combined_final.parquet')  # MES + MGC
train_indices = np.load('config/splits/train_indices.npy')
val_indices = np.load('config/splits/val_indices.npy')

# Create datasets
train_dataset = NHiTSDataset(train_df, train_indices, seq_len=128)
val_dataset = NHiTSDataset(train_df, val_indices, seq_len=128)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4)

# Initialize model
model = NHiTSClassifier(nhits_config).cuda()

# Optimizer and scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5
)

# Loss function
def weighted_ce_loss(logits, targets, weights):
    ce = F.cross_entropy(logits, targets, reduction='none')
    return (ce * weights).mean()
```

#### Training Loop
```python
best_val_loss = float('inf')
patience_counter = 0
patience = 10

for epoch in range(100):
    # Train
    model.train()
    train_loss = 0.0
    for batch in train_loader:
        X = batch['X'].cuda()
        y_1, y_5, y_20 = batch['y_1'].cuda(), batch['y_5'].cuda(), batch['y_20'].cuda()
        w_1, w_5, w_20 = batch['w_1'].cuda(), batch['w_5'].cuda(), batch['w_20'].cuda()
        
        out = model(X)
        loss_1 = weighted_ce_loss(out['logits_1'], y_1, w_1)
        loss_5 = weighted_ce_loss(out['logits_5'], y_5, w_5)
        loss_20 = weighted_ce_loss(out['logits_20'], y_20, w_20)
        loss = loss_1 + loss_5 + loss_20
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    
    # Validation
    model.eval()
    val_loss = 0.0
    all_preds_1, all_preds_5, all_preds_20 = [], [], []
    all_labels_1, all_labels_5, all_labels_20 = [], [], []
    
    with torch.no_grad():
        for batch in val_loader:
            X = batch['X'].cuda()
            y_1, y_5, y_20 = batch['y_1'].cuda(), batch['y_5'].cuda(), batch['y_20'].cuda()
            w_1, w_5, w_20 = batch['w_1'].cuda(), batch['w_5'].cuda(), batch['w_20'].cuda()
            
            out = model(X)
            loss_1 = weighted_ce_loss(out['logits_1'], y_1, w_1)
            loss_5 = weighted_ce_loss(out['logits_5'], y_5, w_5)
            loss_20 = weighted_ce_loss(out['logits_20'], y_20, w_20)
            loss = loss_1 + loss_5 + loss_20
            
            val_loss += loss.item()
            
            # Collect predictions
            preds_1 = torch.argmax(out['logits_1'], dim=1).cpu().numpy()
            preds_5 = torch.argmax(out['logits_5'], dim=1).cpu().numpy()
            preds_20 = torch.argmax(out['logits_20'], dim=1).cpu().numpy()
            
            all_preds_1.extend(preds_1)
            all_preds_5.extend(preds_5)
            all_preds_20.extend(preds_20)
            
            all_labels_1.extend(y_1.cpu().numpy())
            all_labels_5.extend(y_5.cpu().numpy())
            all_labels_20.extend(y_20.cpu().numpy())
    
    val_loss /= len(val_loader)
    
    # Compute metrics
    from sklearn.metrics import accuracy_score, f1_score
    acc_1 = accuracy_score(all_labels_1, all_preds_1)
    f1_1 = f1_score(all_labels_1, all_preds_1, average='macro')
    # Repeat for horizons 5 and 20
    
    print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
    print(f"  1-bar: Acc={acc_1:.3f}, F1={f1_1:.3f}")
    
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'models/nhits_best.pth')
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered")
            break
    
    scheduler.step(val_loss)
```

### Save Model and Predictions

```python
# Save final model
torch.save({
    'model_state_dict': model.state_dict(),
    'config': nhits_config,
    'scaler': scaler,
}, 'models/nhits_final.pth')

# Generate validation predictions (probabilities)
model.eval()
val_probs_1, val_probs_5, val_probs_20 = [], [], []

with torch.no_grad():
    for batch in val_loader:
        X = batch['X'].cuda()
        out = model(X)
        
        probs_1 = F.softmax(out['logits_1'], dim=-1).cpu().numpy()
        probs_5 = F.softmax(out['logits_5'], dim=-1).cpu().numpy()
        probs_20 = F.softmax(out['logits_20'], dim=-1).cpu().numpy()
        
        val_probs_1.append(probs_1)
        val_probs_5.append(probs_5)
        val_probs_20.append(probs_20)

val_probs_1 = np.vstack(val_probs_1)  # [N_val, 3]
val_probs_5 = np.vstack(val_probs_5)
val_probs_20 = np.vstack(val_probs_20)

# Save predictions for Phase 3/4
np.save('predictions/nhits_val_probs_1.npy', val_probs_1)
np.save('predictions/nhits_val_probs_5.npy', val_probs_5)
np.save('predictions/nhits_val_probs_20.npy', val_probs_20)
```

### Backtest N-HiTS Predictions

```python
import vectorbt as vbt

# Load validation data with true prices
val_df = train_df.iloc[val_indices].reset_index(drop=True)

# Use 5-bar predictions for example
pred_classes = np.argmax(val_probs_5, axis=1)  # [N_val]
# Convert back to {-1, 0, +1}
pred_signal = pred_classes - 1  # {0,1,2} → {-1,0,+1}

# Align with close prices
close_prices = val_df['close'].values

# Create entry signals
entries_long = (pred_signal == 1)  # Predict +1 → go long
entries_short = (pred_signal == -1)  # Predict -1 → go short

# Backtest
pf = vbt.Portfolio.from_signals(
    close=close_prices,
    entries=entries_long,
    exits=entries_short,
    freq='5T',  # 5-minute bars
)

sharpe = pf.sharpe_ratio()
max_dd = pf.max_drawdown()
n_trades = pf.trades.count()

print(f"N-HiTS 5-bar Validation Backtest:")
print(f"  Sharpe: {sharpe:.3f}")
print(f"  Max DD: {max_dd:.3%}")
print(f"  Trades: {n_trades}")

# Save tearsheet
fig = pf.plot()
fig.write_html('reports/nhits_5bar_tearsheet.html')
```

### N-HiTS Outputs

**Saved Artifacts:**
- `models/nhits_final.pth` (model weights + config + scaler)
- `models/nhits_scaler.pkl`
- `predictions/nhits_val_probs_{1,5,20}.npy`
- `reports/nhits_metrics.json`
- `reports/nhits_5bar_tearsheet.html`

**Metrics to Report:**
| Horizon | Acc | F1 | Sharpe | Max DD |
|---------|-----|-----|--------|--------|
| 1-bar | ___ | ___ | ___ | ___ |
| 5-bar | ___ | ___ | ___ | ___ |
| 20-bar | ___ | ___ | ___ | ___ |

---

## Model 2: TFT (Temporal Fusion Transformer)

### Model Overview

**TFT** is an attention-based architecture designed for multi-horizon forecasting with built-in interpretability.

**Key Features:**
- LSTM encoder for sequence modeling
- Multi-head attention for temporal fusion
- Supports static, known, and observed inputs
- Variable selection networks (feature importance)
- Interpretable attention weights

**Use Case:** Main workhorse for multi-horizon predictions, strong on heterogeneous features.

### Architecture Configuration

#### Core Hyperparameters

```python
tft_config = {
    # Architecture
    'hidden_size': 128,             # LSTM & attention dimension
    'lstm_layers': 2,               # Number of LSTM layers
    'attention_head_size': 8,       # Number of attention heads
    'dropout': 0.1,
    'hidden_continuous_size': 32,   # Processing continuous features
    
    # Input configuration
    'max_encoder_length': 128,      # Sequence length L
    'max_prediction_length': 20,    # Max forecast horizon
    
    # Feature types
    'time_varying_known_reals': [],       # e.g., hour_sin, hour_cos
    'time_varying_known_categoricals': [],  # e.g., is_rth
    'time_varying_unknown_reals': [],     # All indicators, returns, etc.
    'time_varying_unknown_categoricals': [],
    'static_reals': [],
    'static_categoricals': ['symbol'],    # MES vs MGC
    
    # Output
    'output_size': 3,               # 3 classes per horizon
    'loss': 'CrossEntropy',
    
    # Training
    'learning_rate': 1e-3,
    'batch_size': 128,
    'max_epochs': 50,
    'early_stopping_patience': 7,
}
```

### Model Adaptation for Classification

**PyTorch Forecasting** TFT can be configured for classification by setting:
```python
from pytorch_forecasting import TemporalFusionTransformer

# In training config
loss = CrossEntropy()
output_size = 3  # Classes: {0, 1, 2}
```

**Multi-Horizon Strategy:**
- TFT naturally outputs predictions for multiple time steps ahead
- Set `max_prediction_length = 20`
- Extract predictions at steps 1, 5, 20 as our three horizons

**Modified Output Layer:**
```
TFT Encoder (LSTM + Attention)
    ↓
Decoder outputs [B, H, hidden_size]  # H = 20 time steps
    ↓
Pick step indices: [0, 4, 19]  # 1-bar, 5-bar, 20-bar
    ↓
Classification Head (Linear → 3 classes)
    ↓
logits_1, logits_5, logits_20
```

### Data Preparation for TFT

#### TimeSeriesDataSet Format

PyTorch Forecasting uses a specialized dataset format:

```python
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer

# Prepare DataFrame with required columns
df = pd.read_parquet('data/final/combined_final.parquet')
df['time_idx'] = np.arange(len(df))  # Time index
df['symbol_id'] = df['symbol'].map({'MES': 0, 'MGC': 1})

# Encode labels as targets (one column per horizon)
df['target_1'] = df['label_1'] + 1  # {-1,0,+1} → {0,1,2}
df['target_5'] = df['label_5'] + 1
df['target_20'] = df['label_20'] + 1

# Create training dataset
train_df = df.iloc[train_indices]

training = TimeSeriesDataSet(
    train_df,
    time_idx='time_idx',
    target=['target_1', 'target_5', 'target_20'],  # Multi-target
    group_ids=['symbol'],  # Group by symbol for normalization
    max_encoder_length=128,
    max_prediction_length=20,
    
    # Feature specification
    time_varying_known_reals=['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos'],
    time_varying_unknown_reals=['close', 'volume', 'atr_14', 'rsi_14', ...],  # All indicators
    static_categoricals=['symbol_id'],
    
    # Normalization
    target_normalizer=None,  # Classification, no normalization
    scalers={
        'hour_sin': GroupNormalizer(transformation=None),  # Already [-1,1]
        'close': GroupNormalizer(),  # Normalize per symbol
        # ... specify for each feature
    },
    
    # Add sample weights (custom handling needed)
    # TFT doesn't natively support sample weights, may need to modify loss
)

# Create validation dataset (use same scaling as training)
validation = TimeSeriesDataSet.from_dataset(training, val_df, predict=True, stop_randomization=True)

# DataLoaders
train_loader = training.to_dataloader(train=True, batch_size=128, num_workers=4)
val_loader = validation.to_dataloader(train=False, batch_size=256, num_workers=4)
```

### Training Procedure

#### Setup
```python
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

# Initialize TFT model
tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=1e-3,
    hidden_size=128,
    attention_head_size=8,
    dropout=0.1,
    hidden_continuous_size=32,
    output_size=3,  # 3 classes
    loss=torch.nn.CrossEntropyLoss(),  # Classification loss
    log_interval=10,
    reduce_on_plateau_patience=5,
)

# PyTorch Lightning Trainer
trainer = Trainer(
    max_epochs=50,
    accelerator='gpu',
    devices=1,
    gradient_clip_val=0.1,
    callbacks=[
        EarlyStopping(monitor='val_loss', patience=7, mode='min'),
        ModelCheckpoint(
            monitor='val_loss',
            dirpath='models/',
            filename='tft-{epoch:02d}-{val_loss:.4f}',
            save_top_k=1,
            mode='min',
        ),
    ],
)

# Train
trainer.fit(
    tft,
    train_dataloaders=train_loader,
    val_dataloaders=val_loader,
)
```

**Note on Multi-Horizon:**
- TFT will output a sequence of length 20 (max_prediction_length)
- For classification, each step outputs 3-class logits
- During evaluation, extract logits at step indices [0, 4, 19] for our horizons

#### Custom Multi-Horizon Loss

If using custom loss for multi-horizon weighting:

```python
class MultiHorizonCELoss(nn.Module):
    def __init__(self, horizon_indices=[0, 4, 19], weights=[1.0, 1.0, 1.0]):
        super().__init__()
        self.horizon_indices = horizon_indices
        self.weights = weights
        self.ce = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, predictions, targets):
        # predictions: [B, H, 3], H=20 steps
        # targets: [B, 3] for three horizons
        
        loss = 0.0
        for i, (h_idx, w) in enumerate(zip(self.horizon_indices, self.weights)):
            pred_h = predictions[:, h_idx, :]  # [B, 3]
            target_h = targets[:, i]  # [B]
            loss += w * self.ce(pred_h, target_h).mean()
        
        return loss
```

### Interpretability: Feature Importance

**TFT Variable Selection:**
After training, extract feature importances:

```python
interpretation = tft.interpret_output(
    val_loader.dataset,
    mode='raw',
    return_x=True,
)

# Variable importance (which features matter most)
feature_importance = interpretation['feature_importance']
print("Top 10 Features:")
print(feature_importance.nlargest(10))

# Attention patterns (which past timesteps are attended to)
attention_weights = interpretation['attention']

# Save for later analysis
import pickle
with open('reports/tft_interpretation.pkl', 'wb') as f:
    pickle.dump(interpretation, f)
```

### Save Model and Predictions

```python
# Save best model
best_model_path = trainer.checkpoint_callback.best_model_path
tft_best = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

# Generate validation predictions
predictions = tft_best.predict(val_loader, return_x=True)

# predictions shape: [N_val, max_pred_len, 3] = [N_val, 20, 3]
# Extract horizons
pred_probs = predictions.cpu().numpy()
val_probs_1 = pred_probs[:, 0, :]   # [N_val, 3]
val_probs_5 = pred_probs[:, 4, :]   # [N_val, 3]
val_probs_20 = pred_probs[:, 19, :]  # [N_val, 3]

# Apply softmax if not already applied
val_probs_1 = softmax(val_probs_1, axis=-1)
val_probs_5 = softmax(val_probs_5, axis=-1)
val_probs_20 = softmax(val_probs_20, axis=-1)

# Save
np.save('predictions/tft_val_probs_1.npy', val_probs_1)
np.save('predictions/tft_val_probs_5.npy', val_probs_5)
np.save('predictions/tft_val_probs_20.npy', val_probs_20)

# Save model
torch.save({
    'model_state_dict': tft_best.state_dict(),
    'config': tft_config,
}, 'models/tft_final.pth')
```

### Backtest TFT Predictions

```python
# Similar to N-HiTS backtest
pred_classes_5 = np.argmax(val_probs_5, axis=1) - 1  # {0,1,2} → {-1,0,+1}

entries_long = (pred_classes_5 == 1)
entries_short = (pred_classes_5 == -1)

pf = vbt.Portfolio.from_signals(
    close=val_df['close'].values,
    entries=entries_long,
    exits=entries_short,
    freq='5T',
)

sharpe = pf.sharpe_ratio()
max_dd = pf.max_drawdown()

print(f"TFT 5-bar Validation Backtest:")
print(f"  Sharpe: {sharpe:.3f}")
print(f"  Max DD: {max_dd:.3%}")

# Save tearsheet
pf.plot().write_html('reports/tft_5bar_tearsheet.html')
```

### TFT Outputs

**Saved Artifacts:**
- `models/tft_final.pth`
- `predictions/tft_val_probs_{1,5,20}.npy`
- `reports/tft_interpretation.pkl`
- `reports/tft_metrics.json`
- `reports/tft_5bar_tearsheet.html`

**Metrics to Report:**
| Horizon | Acc | F1 | Sharpe | Max DD |
|---------|-----|-----|--------|--------|
| 1-bar | ___ | ___ | ___ | ___ |
| 5-bar | ___ | ___ | ___ | ___ |
| 20-bar | ___ | ___ | ___ | ___ |

---

## Model 3: PatchTST (Patch Time Series Transformer)

### Model Overview

**PatchTST** is a state-of-the-art Transformer that operates on patches (segments) of the time series rather than individual timesteps.

**Key Features:**
- Patch-based tokenization (reduces sequence length)
- Channel-independent processing (each feature as separate channel)
- Self-attention over patch tokens
- Excellent long-term dependency capture
- ICLR 2023 state-of-the-art on many benchmarks

**Use Case:** Modern transformer for long horizons (especially 20-bar), captures complex temporal patterns.

### Architecture Configuration

#### Core Hyperparameters

```python
patchtst_config = {
    # Patching
    'patch_len': 16,                # Length of each patch
    'stride': 8,                    # Stride for patch extraction
    
    # Transformer
    'd_model': 128,                 # Model dimension
    'n_heads': 8,                   # Number of attention heads
    'n_layers': 3,                  # Transformer encoder layers
    'd_ff': 256,                    # Feedforward network hidden dim
    'dropout': 0.1,
    'activation': 'gelu',
    
    # Input/Output
    'seq_len': 128,                 # Input sequence length L
    'pred_len': 20,                 # Prediction horizon
    'n_features': 50,               # Number of input features F
    
    # Normalization
    'use_revin': True,              # Reversible Instance Normalization
    
    # Training
    'learning_rate': 1e-4,
    'batch_size': 32,
    'max_epochs': 100,
    'early_stopping_patience': 10,
}
```

### Patch Mechanism Explained

**Concept:**
- Instead of feeding each timestep to Transformer (expensive), split sequence into patches
- Example: `seq_len=128`, `patch_len=16`, `stride=8`
  - Patches: [0:16], [8:24], [16:32], ... → ~15 patches
- Each patch becomes one "token" for the Transformer

**Benefits:**
- Reduces sequence length from 128 to 15 (much faster attention)
- Captures local patterns within patches
- Self-attention connects patches across time

### Model Adaptation for Classification

**Modified Architecture:**
```
Input [B, L, F] where L=128, F=50
    ↓
Patchify per channel
    ↓ (channel-independent)
For each feature channel:
    Patches [B, N_patches, patch_len]
    ↓
    Embedding [B, N_patches, d_model]
    ↓
    Transformer Encoder [B, N_patches, d_model]
    ↓
    Pooling (mean or CLS token) → h_channel [B, d_model]
    ↓
Concatenate all channels → h [B, F * d_model]
    ↓
Projection → h_proj [B, H_mid]
    ↓
Multi-horizon heads
    ├── Head_1 → logits_1 [B, 3]
    ├── Head_5 → logits_5 [B, 3]
    └── Head_20 → logits_20 [B, 3]
```

### Implementation Sketch

```python
class PatchTSTClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.patch_len = config['patch_len']
        self.stride = config['stride']
        self.d_model = config['d_model']
        self.n_features = config['n_features']
        
        # Calculate number of patches
        self.n_patches = (config['seq_len'] - self.patch_len) // self.stride + 1
        
        # Patch embedding (per channel)
        self.patch_embedding = nn.Linear(self.patch_len, self.d_model)
        
        # Positional encoding
        self.positional_encoding = nn.Parameter(
            torch.randn(1, self.n_patches, self.d_model)
        )
        
        # Transformer encoder (shared across channels if channel-independent)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=config['n_heads'],
            dim_feedforward=config['d_ff'],
            dropout=config['dropout'],
            activation=config['activation'],
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config['n_layers'],
        )
        
        # Projection after pooling
        H_mid = 256
        self.projection = nn.Sequential(
            nn.Linear(self.n_features * self.d_model, H_mid),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        
        # Classification heads
        self.head_1 = nn.Linear(H_mid, 3)
        self.head_5 = nn.Linear(H_mid, 3)
        self.head_20 = nn.Linear(H_mid, 3)
        
        # RevIN (optional)
        if config['use_revin']:
            self.revin = RevIN(self.n_features)
    
    def patchify(self, x):
        # x: [B, L, F]
        B, L, F = x.shape
        patches = []
        for i in range(0, L - self.patch_len + 1, self.stride):
            patch = x[:, i:i+self.patch_len, :]  # [B, patch_len, F]
            patches.append(patch)
        patches = torch.stack(patches, dim=1)  # [B, N_patches, patch_len, F]
        return patches
    
    def forward(self, x):
        # x: [B, L, F]
        B, L, F = x.shape
        
        # RevIN normalization
        if hasattr(self, 'revin'):
            x = self.revin(x, mode='norm')
        
        # Patchify: [B, N_patches, patch_len, F]
        patches = self.patchify(x)
        N_patches = patches.shape[1]
        
        # Process each channel independently
        channel_embeddings = []
        for f_idx in range(F):
            # Patches for this channel: [B, N_patches, patch_len]
            channel_patches = patches[:, :, :, f_idx]
            
            # Embed patches: [B, N_patches, d_model]
            channel_embed = self.patch_embedding(channel_patches)
            
            # Add positional encoding
            channel_embed = channel_embed + self.positional_encoding
            
            # Transformer encoder
            channel_encoded = self.transformer(channel_embed)  # [B, N_patches, d_model]
            
            # Pool (mean over patches)
            channel_pooled = channel_encoded.mean(dim=1)  # [B, d_model]
            
            channel_embeddings.append(channel_pooled)
        
        # Concatenate all channels: [B, F * d_model]
        h = torch.cat(channel_embeddings, dim=1)
        
        # Projection
        h_proj = self.projection(h)  # [B, H_mid]
        
        # Multi-horizon heads
        logits_1 = self.head_1(h_proj)
        logits_5 = self.head_5(h_proj)
        logits_20 = self.head_20(h_proj)
        
        return {
            'logits_1': logits_1,
            'logits_5': logits_5,
            'logits_20': logits_20,
        }

class RevIN(nn.Module):
    """Reversible Instance Normalization"""
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
    
    def forward(self, x, mode='norm'):
        if mode == 'norm':
            # x: [B, L, F]
            self.mean = x.mean(dim=1, keepdim=True)  # [B, 1, F]
            self.std = x.std(dim=1, keepdim=True) + self.eps
            return (x - self.mean) / self.std
        elif mode == 'denorm':
            return x * self.std + self.mean
```

### Data Preparation for PatchTST

Same as N-HiTS, use `NHiTSDataset` or similar:

```python
# Can reuse NHiTSDataset class
train_dataset = NHiTSDataset(train_df, train_indices, seq_len=128)
val_dataset = NHiTSDataset(val_df, val_indices, seq_len=128)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
```

### Training Procedure

```python
# Initialize model
model = PatchTSTClassifier(patchtst_config).cuda()

# Optimizer (lower LR for transformer)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)

# Training loop (same structure as N-HiTS)
best_val_loss = float('inf')
patience_counter = 0
patience = 10

for epoch in range(100):
    model.train()
    train_loss = 0.0
    
    for batch in train_loader:
        X = batch['X'].cuda()
        y_1, y_5, y_20 = batch['y_1'].cuda(), batch['y_5'].cuda(), batch['y_20'].cuda()
        w_1, w_5, w_20 = batch['w_1'].cuda(), batch['w_5'].cuda(), batch['w_20'].cuda()
        
        out = model(X)
        loss_1 = weighted_ce_loss(out['logits_1'], y_1, w_1)
        loss_5 = weighted_ce_loss(out['logits_5'], y_5, w_5)
        loss_20 = weighted_ce_loss(out['logits_20'], y_20, w_20)
        loss = loss_1 + loss_5 + loss_20
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
        optimizer.step()
        
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    
    # Validation (same as N-HiTS)
    model.eval()
    val_loss, val_metrics = evaluate_model(model, val_loader)
    
    print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
    
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'models/patchtst_best.pth')
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping")
            break
    
    scheduler.step()
```

### Save Model and Predictions

```python
# Load best model
model.load_state_dict(torch.load('models/patchtst_best.pth'))

# Generate validation predictions
model.eval()
val_probs_1, val_probs_5, val_probs_20 = [], [], []

with torch.no_grad():
    for batch in val_loader:
        X = batch['X'].cuda()
        out = model(X)
        
        probs_1 = F.softmax(out['logits_1'], dim=-1).cpu().numpy()
        probs_5 = F.softmax(out['logits_5'], dim=-1).cpu().numpy()
        probs_20 = F.softmax(out['logits_20'], dim=-1).cpu().numpy()
        
        val_probs_1.append(probs_1)
        val_probs_5.append(probs_5)
        val_probs_20.append(probs_20)

val_probs_1 = np.vstack(val_probs_1)
val_probs_5 = np.vstack(val_probs_5)
val_probs_20 = np.vstack(val_probs_20)

# Save
np.save('predictions/patchtst_val_probs_1.npy', val_probs_1)
np.save('predictions/patchtst_val_probs_5.npy', val_probs_5)
np.save('predictions/patchtst_val_probs_20.npy', val_probs_20)

# Save model
torch.save({
    'model_state_dict': model.state_dict(),
    'config': patchtst_config,
}, 'models/patchtst_final.pth')
```

### Backtest PatchTST Predictions

```python
# Backtest (same as previous models)
pred_classes_5 = np.argmax(val_probs_5, axis=1) - 1

entries_long = (pred_classes_5 == 1)
entries_short = (pred_classes_5 == -1)

pf = vbt.Portfolio.from_signals(
    close=val_df['close'].values,
    entries=entries_long,
    exits=entries_short,
    freq='5T',
)

sharpe = pf.sharpe_ratio()
max_dd = pf.max_drawdown()

print(f"PatchTST 5-bar Validation Backtest:")
print(f"  Sharpe: {sharpe:.3f}")
print(f"  Max DD: {max_dd:.3%}")

pf.plot().write_html('reports/patchtst_5bar_tearsheet.html')
```

### PatchTST Outputs

**Saved Artifacts:**
- `models/patchtst_final.pth`
- `predictions/patchtst_val_probs_{1,5,20}.npy`
- `reports/patchtst_metrics.json`
- `reports/patchtst_5bar_tearsheet.html`

**Metrics to Report:**
| Horizon | Acc | F1 | Sharpe | Max DD |
|---------|-----|-----|--------|--------|
| 1-bar | ___ | ___ | ___ | ___ |
| 5-bar | ___ | ___ | ___ | ___ |
| 20-bar | ___ | ___ | ___ | ___ |

---

## Phase 2 Deliverables Summary

### Primary Outputs

For each model (N-HiTS, TFT, PatchTST):

1. **Trained Model Weights**
   - `models/{model}_final.pth`
   - Includes: state_dict, config, scaler (if applicable)

2. **Validation Predictions (Probabilities)**
   - `predictions/{model}_val_probs_1.npy` [N_val, 3]
   - `predictions/{model}_val_probs_5.npy` [N_val, 3]
   - `predictions/{model}_val_probs_20.npy` [N_val, 3]
   - These are crucial inputs for Phase 3/4 (ensemble stacking)

3. **Metrics Reports**
   - `reports/{model}_metrics.json`
   - Contains: Accuracy, F1, Sharpe, Max DD for all horizons

4. **Backtest Tearsheets**
   - `reports/{model}_{horizon}_tearsheet.html`
   - Vectorbt portfolio visualization

5. **Training Logs**
   - `logs/{model}_training.log`
   - Epoch-by-epoch loss, metrics, learning rate

### Comparative Analysis

**Create comparison report:**
```
reports/phase2_model_comparison.md
```

**Contents:**
- Side-by-side metrics table for all models and horizons
- Best model per horizon
- Correlation between model predictions (do they complement each other?)
- Computational cost comparison (training time, inference speed)

**Example table:**
| Model | Horizon | Acc | F1 | Sharpe | Max DD | Train Time |
|-------|---------|-----|-----|--------|--------|------------|
| N-HiTS | 1-bar | 0.52 | 0.38 | 0.45 | 12% | 2h |
| N-HiTS | 5-bar | 0.55 | 0.42 | 0.62 | 9% | 2h |
| N-HiTS | 20-bar | 0.53 | 0.40 | 0.58 | 11% | 2h |
| TFT | 1-bar | 0.54 | 0.40 | 0.51 | 10% | 8h |
| TFT | 5-bar | 0.57 | 0.44 | 0.68 | 8% | 8h |
| TFT | 20-bar | 0.56 | 0.43 | 0.64 | 9% | 8h |
| PatchTST | 1-bar | 0.53 | 0.39 | 0.48 | 11% | 6h |
| PatchTST | 5-bar | 0.56 | 0.43 | 0.65 | 9% | 6h |
| PatchTST | 20-bar | 0.58 | 0.46 | 0.72 | 7% | 6h |

---

## Implementation Checklist

### Pre-Phase Tasks
- [ ] Verify Phase 1 outputs exist and are valid
- [ ] Set up model directories (`models/`, `predictions/`, `reports/`, `logs/`)
- [ ] Install required libraries (torch, neuralforecast, pytorch-forecasting, vectorbt)
- [ ] Configure GPU environment and verify CUDA availability

### N-HiTS Module
- [ ] Implement `NHiTSClassifier` with multi-horizon heads
- [ ] Create `NHiTSDataset` class with proper windowing
- [ ] Compute feature scaling (StandardScaler) on train set only
- [ ] Set up training loop with early stopping
- [ ] Train N-HiTS model (monitor train/val metrics)
- [ ] Generate validation predictions (probabilities)
- [ ] Run backtest on validation set for all horizons
- [ ] Save model weights, scaler, and predictions
- [ ] Generate metrics report and tearsheets

### TFT Module
- [ ] Prepare data in PyTorch Forecasting `TimeSeriesDataSet` format
- [ ] Configure TFT for classification (CrossEntropy loss, output_size=3)
- [ ] Set up feature type assignments (known, unknown, static)
- [ ] Initialize TFT model from dataset
- [ ] Set up PyTorch Lightning Trainer with callbacks
- [ ] Train TFT model
- [ ] Extract validation predictions at horizon indices [0, 4, 19]
- [ ] Apply softmax to convert logits to probabilities
- [ ] Run backtest on validation set
- [ ] Extract feature importance and attention interpretations
- [ ] Save model, predictions, and interpretation artifacts

### PatchTST Module
- [ ] Implement `PatchTSTClassifier` with patching mechanism
- [ ] Implement RevIN normalization layer
- [ ] Reuse `NHiTSDataset` or create equivalent
- [ ] Set up training loop with gradient clipping
- [ ] Train PatchTST model
- [ ] Generate validation predictions
- [ ] Run backtest on validation set
- [ ] Save model weights and predictions

### Cross-Model Analysis
- [ ] Load all validation predictions (9 files total: 3 models × 3 horizons)
- [ ] Compute correlation matrix between model predictions
- [ ] Identify which models are complementary (low correlation = good for ensembling)
- [ ] Create comparative metrics table
- [ ] Identify best model per horizon
- [ ] Document strengths/weaknesses of each model

### Post-Phase Tasks
- [ ] Archive training logs and checkpoints
- [ ] Update project README with Phase 2 results
- [ ] Verify all validation predictions align (same sample order, lengths)
- [ ] Prepare validation probabilities for Phase 3 (purged CV) and Phase 4 (stacking)

---

## Notes and Considerations

### Model Selection and Hyperparameter Tuning

**Initial Configs:**
- Start with documented default hyperparameters for each model
- These are "sane defaults" that should work reasonably well

**If Performance is Poor:**
- Use Optuna for hyperparameter search:
  - Learning rate: [1e-5, 1e-2]
  - Hidden sizes: [64, 128, 256, 512]
  - Dropout: [0.0, 0.1, 0.2, 0.3]
  - Batch size: [32, 64, 128, 256]
- Focus on validation Sharpe + F1 as optimization target
- Budget 2-5× training time for hyperparam search

**Diminishing Returns:**
- Squeezing extra 2-3% F1 via extensive tuning has less value than moving to Phase 3/4
- "Good enough" > "perfect" at this stage

### Handling Class Imbalance

**Problem:**
- Label distribution is imbalanced (0 label is ~50-70%)
- Models may over-predict class 1 (neutral)

**Solutions:**
1. **Sample Weights (already implemented)**
   - Assign higher weights to ±1 labels in loss function
   - Effective and simple

2. **Focal Loss (alternative)**
   - Downweights easy examples, focuses on hard ones
   - Can replace standard CrossEntropy

3. **Class Weights in Loss**
   ```python
   class_weights = torch.FloatTensor([2.0, 1.0, 2.0])  # Up-weight ±1 classes
   loss = F.cross_entropy(logits, targets, weight=class_weights)
   ```

4. **Threshold Tuning (Phase 6)**
   - Adjust decision threshold for argmax
   - E.g., only predict +1 if P(+1) > 0.5 (instead of just > P(0) and P(-1))

### Computational Considerations

**Training Time Estimates:**
- N-HiTS: 2-4 hours (fast MLP)
- TFT: 8-12 hours (attention + LSTM)
- PatchTST: 6-10 hours (transformer, but patching helps)

**GPU Memory:**
- Batch size tuning may be needed
- If OOM errors:
  - Reduce batch size
  - Reduce sequence length
  - Reduce hidden sizes
  - Use gradient accumulation

**Parallelization:**
- Train three models in parallel if you have 3× GPUs
- Or train sequentially (safer, less complexity)

### Overfitting Prevention

**Signs of Overfitting:**
- Train loss keeps decreasing, val loss plateaus or increases
- Train F1/Sharpe much higher than validation

**Mitigations:**
- Dropout (already in configs)
- Weight decay / L2 regularization
- Early stopping (critical)
- Data augmentation (less common in TS, but can add noise)

### Feature Engineering Feedback

**During Training:**
- Monitor which features TFT considers important
- If some features are consistently ignored, consider dropping them
- This can feed back into Phase 1 for refinement

### Known Issues and Edge Cases

**TFT Multi-Target Handling:**
- PyTorch Forecasting TFT expects single target by default
- May need to train three separate TFTs (one per horizon) if multi-target is problematic
- Alternative: use custom TFT implementation or modify library

**PatchTST Channel Independence:**
- Each feature treated as separate channel can be memory-intensive if F is large (50+)
- Consider: reduce feature count or use shared processing across channels (deviation from paper)

**RevIN with Classification:**
- RevIN designed for regression (denormalization at output)
- For classification, we only use normalization step (no denorm)

### Extension Ideas

**Multi-Asset Joint Training:**
- Current setup: MES and MGC trained together (symbol_id feature)
- Alternative: train separate models per symbol
- Tradeoff: joint training shares knowledge, separate models specialize

**Ensemble Within Model:**
- Instead of one N-HiTS, train 3-5 with different random seeds
- Average their predictions (mini-ensemble before stacking)
- Reduces variance, improves robustness

**Auxiliary Tasks:**
- Add auxiliary regression head: predict actual return magnitude
- Multi-task learning can improve shared representations

---

## Success Criteria Summary

Phase 2 is complete and successful when:

1. ✅ All three models (N-HiTS, TFT, PatchTST) trained successfully
2. ✅ Each model achieves validation F1 > 0.35 on at least one horizon
3. ✅ Each model achieves validation Sharpe > 0.3 on at least one horizon
4. ✅ Validation predictions saved in consistent format for all models/horizons
5. ✅ No severe overfitting (train/val gap < 20%)
6. ✅ Comprehensive metrics and backtest tearsheets generated
7. ✅ Comparative analysis completed
8. ✅ All outputs documented and organized

**Proceed to Phase 3** (Cross-Validation) only after all criteria are met.

---

**End of Phase 2 Specification**
