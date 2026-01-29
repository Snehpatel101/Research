# ML Factory User Guide

**A complete guide to training ML trading ensembles using Google Colab with VS Code.**

---

## Table of Contents

1. [Quick Start (5 minutes)](#quick-start-5-minutes)
2. [Setup Options](#setup-options)
   - [Option A: VS Code + Colab Extension](#option-a-vs-code--colab-extension-recommended)
   - [Option B: Web-Based Google Colab](#option-b-web-based-google-colab)
3. [Configuration Reference](#configuration-reference)
   - [Quick Start Configs](#quick-start-configs)
   - [Data Configuration](#data-configuration)
   - [Model Selection](#model-selection)
   - [Optimization Settings](#optimization-settings)
   - [Ensemble Settings](#ensemble-settings)
   - [Evaluation Settings](#evaluation-settings)
4. [Complete Config Reference Table](#complete-config-reference-table)
5. [Tips and Best Practices](#tips-and-best-practices)
6. [Troubleshooting](#troubleshooting)

---

## Quick Start (5 minutes)

### Prerequisites
- VS Code installed
- Google account (free tier works)
- Your OHLCV data file (`.parquet` or `.csv`)

### Steps

1. **Install the Google Colab extension in VS Code:**
   - Press `Ctrl+Shift+X` (or `Cmd+Shift+X` on Mac)
   - Search for "Google Colab"
   - Install the official extension by Google

2. **Open the notebook:**
   - Open `notebooks/ml_factory_colab.ipynb` in VS Code

3. **Connect to Colab runtime:**
   - Click "Select Kernel" (top-right)
   - Select "Colab" → "New Colab Server"
   - Sign in with Google
   - Choose "T4 GPU" for best results

4. **Configure and run:**
   - Edit Cell 2 with your settings (see below)
   - Run all cells (`Shift+Enter` through each cell)

---

## Setup Options

### Option A: VS Code + Colab Extension (Recommended)

The Google Colab VS Code extension lets you use Colab's free GPUs directly from VS Code.

#### Installation

1. **Open VS Code Extensions** (`Ctrl+Shift+X`)
2. **Search** for "Google Colab"
3. **Install** the official extension by Google
4. **Restart VS Code** if prompted

#### Connecting to a Runtime

1. Open any `.ipynb` notebook file
2. Click **"Select Kernel"** (top-right corner)
3. Select **"Colab"** → **"New Colab Server"**
4. **Sign in** with your Google account
5. **Choose your runtime:**

| Runtime | VRAM | Best For | Availability |
|---------|------|----------|--------------|
| **T4 GPU** | 16GB | Most use cases | Free tier |
| **A100 GPU** | 40GB | Large transformers | Colab Pro |
| **L4 GPU** | 24GB | Production training | Colab Pro |
| **CPU** | N/A | Testing only | Always |

#### Data Location

The notebook defaults to using data from the repo's `data/` folder:

**Default (Recommended):**
```python
# Uses repo's data folder - upload your data there before running
DATA_PATH = "/content/ml_factory/data/clean/MES_5m_clean.parquet"
```

**Alternative Options:**

**Option 1: Upload directly (small files < 100MB)**
```python
from google.colab import files
uploaded = files.upload()  # Opens file picker
DATA_PATH = "/content/your_data.parquet"
```

**Option 2: Download from URL**
```python
!wget -O /content/data.parquet "https://your-storage.com/data.parquet"
DATA_PATH = "/content/data.parquet"
```

**Option 3: Google Drive (Web Colab only)**
```python
# Uncomment drive.mount() in Cell 1 first
DATA_PATH = "/content/drive/MyDrive/data/MES_5min.parquet"
```

**Note:** Google Drive mounting does NOT work in VS Code extension.

#### Key Commands

| Action | VS Code Command |
|--------|-----------------|
| Run cell | `Shift+Enter` |
| Run all cells | `Ctrl+Shift+Enter` |
| Disconnect runtime | `Ctrl+Shift+P` → "Colab: Remove Server" |
| Sign out | `Ctrl+Shift+P` → "Colab: Sign Out" |

---

### Option B: Web-Based Google Colab

Traditional Colab in your browser with full Google Drive integration.

#### Steps

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. **File** → **Open notebook** → **GitHub**
3. Enter: `https://github.com/Snehpatel101/Research`
4. Select `notebooks/ml_factory_colab.ipynb`
5. **Runtime** → **Change runtime type** → **T4 GPU**
6. Run all cells

#### Google Drive Integration (Web Only)

The web version supports automatic Drive mounting:
```python
from google.colab import drive
drive.mount('/content/drive')

# Access files from your Drive
DATA_PATH = "/content/drive/MyDrive/data/MES_5min.parquet"
```

---

## Configuration Reference

### Quick Start Configs

#### Beginner: Minimal Config
```python
# Just change these two lines!
SYMBOL = "MES"
DATA_PATH = "/content/ml_factory/data/clean/MES_5m_clean.parquet"

# Everything else uses sensible defaults
MODELS = ["xgboost", "lightgbm"]
OPTIMIZE_FOR = "sharpe_ratio"
HORIZONS = [20]
BUILD_ENSEMBLE = False
OPTUNA_TRIALS = 25
LABELING_METHOD = "triple_barrier"
FEATURE_FAMILIES = ["price", "momentum", "volatility"]
RUN_BACKTEST = True
POSITION_SIZING = "fixed"
EXPERIMENT_NAME = "quick_test"
RANDOM_SEED = 42
SAVE_RESULTS = True
```
**Runtime:** ~5-15 minutes

---

#### Intermediate: Balanced Config
```python
SYMBOL = "MES"
DATA_PATH = "/content/ml_factory/data/clean/MES_5m_clean.parquet"

MODELS = ["xgboost", "lightgbm", "catboost", "lstm"]
OPTIMIZE_FOR = "sharpe_ratio"
HORIZONS = [5, 10, 20]
BUILD_ENSEMBLE = True
META_LEARNER = "ridge_meta"
OPTUNA_TRIALS = 50
LABELING_METHOD = "triple_barrier"
FEATURE_FAMILIES = ["price", "momentum", "volatility", "volume", "trend"]
RUN_BACKTEST = True
POSITION_SIZING = "confidence"
EXPERIMENT_NAME = "balanced_run"
RANDOM_SEED = 42
SAVE_RESULTS = True
```
**Runtime:** ~30-90 minutes (GPU required for LSTM)

---

#### Advanced: Full Production Config
```python
SYMBOL = "MES"
DATA_PATH = "/content/ml_factory/data/clean/MES_5m_clean.parquet"

MODELS = [
    "xgboost", "lightgbm", "catboost",
    "lstm", "gru", "tcn",
    "patchtst", "itransformer"
]
OPTIMIZE_FOR = "sharpe_ratio"
HORIZONS = [5, 10, 15, 20]
BUILD_ENSEMBLE = True
META_LEARNER = "ridge_meta"
OPTUNA_TRIALS = 100
LABELING_METHOD = "triple_barrier"
FEATURE_FAMILIES = [
    "price", "momentum", "volatility", "volume",
    "trend", "microstructure", "regime"
]
RUN_BACKTEST = True
POSITION_SIZING = "kelly"
EXPERIMENT_NAME = "production_run"
RANDOM_SEED = 42
SAVE_RESULTS = True
```
**Runtime:** ~2-6 hours (GPU required)

---

### Data Configuration

#### SYMBOL
| Property | Value |
|----------|-------|
| **Type** | String |
| **Default** | `"MES"` |
| **Required** | Yes |

The trading symbol for your data. Used for labeling outputs.

**Examples:** `"MES"`, `"ES"`, `"NQ"`, `"SPY"`, `"AAPL"`

---

#### DATA_PATH
| Property | Value |
|----------|-------|
| **Type** | String (file path) |
| **Default** | None |
| **Required** | Yes |

Path to your OHLCV data file. Must contain columns: `open`, `high`, `low`, `close`, `volume`.

**Examples:**
```python
# Repo's data folder (default)
DATA_PATH = "/content/ml_factory/data/clean/MES_5m_clean.parquet"

# Uploaded to session
DATA_PATH = "/content/my_data.parquet"

# Google Drive (web Colab only)
DATA_PATH = "/content/drive/MyDrive/data/MES_5min.parquet"
```

---

#### FEATURE_FAMILIES
| Property | Value |
|----------|-------|
| **Type** | List of strings |
| **Default** | `["price", "momentum", "volatility", "volume", "trend"]` |

Which technical indicator categories to generate.

| Family | # Features | Description |
|--------|-----------|-------------|
| `"price"` | 12 | Returns, log prices, price ratios |
| `"momentum"` | 23 | RSI, MACD, Stochastic, ROC |
| `"volatility"` | 25 | ATR, Bollinger Bands, Keltner |
| `"volume"` | 15 | OBV, VWAP, volume ratios |
| `"trend"` | 6 | ADX, Aroon, MA crossovers |
| `"microstructure"` | 15 | Order flow indicators |
| `"regime"` | 9 | Market regime detection |
| `"wavelet"` | 15 | Wavelet decomposition |

---

#### LABELING_METHOD
| Property | Value |
|----------|-------|
| **Type** | String |
| **Default** | `"triple_barrier"` |

How to create buy/sell signals from price data.

| Method | Description |
|--------|-------------|
| `"triple_barrier"` | Uses profit target + stop loss + time limit (recommended) |
| `"directional"` | Simple future price direction |
| `"threshold"` | Requires minimum return |

---

#### HORIZONS
| Property | Value |
|----------|-------|
| **Type** | List of integers |
| **Default** | `[5, 10, 15, 20]` |

How many bars ahead to predict. Each creates a separate model.

**Examples:**
```python
HORIZONS = [20]           # Single horizon (fastest)
HORIZONS = [5, 10, 20]    # Multiple horizons
```

---

### Model Selection

#### MODELS
| Property | Value |
|----------|-------|
| **Type** | List of strings |
| **Default** | `["xgboost", "lightgbm"]` |

Which ML models to train.

##### Fast Models (No GPU needed, 1-3 min each)
| Model | Description |
|-------|-------------|
| `"xgboost"` | Gradient boosting - excellent all-around |
| `"lightgbm"` | Fast gradient boosting - great for large data |
| `"catboost"` | Gradient boosting with categorical handling |

##### Medium Models (GPU helpful, 5-15 min each)
| Model | Description |
|-------|-------------|
| `"lstm"` | Long Short-Term Memory neural network |
| `"gru"` | Gated Recurrent Unit - simpler than LSTM |
| `"nbeats"` | Neural Basis Expansion - interpretable |

##### Slow Models (GPU recommended, 15-45 min each)
| Model | Description |
|-------|-------------|
| `"tcn"` | Temporal Convolutional Network |
| `"inception_time"` | CNN-based time series model |
| `"resnet_1d"` | 1D Residual Network |

##### GPU-Heavy Models (GPU required, 30-90 min each)
| Model | Description |
|-------|-------------|
| `"patchtst"` | Patch Time Series Transformer - state-of-the-art |
| `"itransformer"` | Inverted Transformer - novel architecture |
| `"tft"` | Temporal Fusion Transformer - interpretable |

---

### Optimization Settings

#### OPTIMIZE_FOR
| Property | Value |
|----------|-------|
| **Type** | String |
| **Default** | `"sharpe_ratio"` |

What metric to optimize during training.

| Metric | Description | Best For |
|--------|-------------|----------|
| `"sharpe_ratio"` | Risk-adjusted return | Trading (recommended) |
| `"sortino_ratio"` | Downside-risk adjusted | Asymmetric risk |
| `"profit_factor"` | Gross profit / loss | Simple profitability |
| `"f1_weighted"` | Classification accuracy | Balanced prediction |
| `"accuracy"` | % correct | Simple baseline |

---

#### OPTUNA_TRIALS
| Property | Value |
|----------|-------|
| **Type** | Integer |
| **Default** | `50` |

How many hyperparameter combinations to try.

| Trials | Quality | Time |
|--------|---------|------|
| 25 | Quick test | ~1x |
| 50 | Good balance | ~2x |
| 100 | Production | ~4x |

---

### Ensemble Settings

#### BUILD_ENSEMBLE
| Property | Value |
|----------|-------|
| **Type** | Boolean |
| **Default** | `True` |

Whether to combine multiple models into one ensemble.

---

#### META_LEARNER
| Property | Value |
|----------|-------|
| **Type** | String |
| **Default** | `"ridge_meta"` |

How to combine model predictions.

| Meta-Learner | Description |
|--------------|-------------|
| `"ridge_meta"` | Ridge regression (recommended) |
| `"mlp_meta"` | Neural network |
| `"xgboost_meta"` | XGBoost |
| `"calibrated_meta"` | Probability calibrated |

---

### Evaluation Settings

#### RUN_BACKTEST
| Property | Value |
|----------|-------|
| **Type** | Boolean |
| **Default** | `True` |

Whether to run simulated trading after training.

---

#### POSITION_SIZING
| Property | Value |
|----------|-------|
| **Type** | String |
| **Default** | `"fixed"` |

How to size trades.

| Method | Risk | Description |
|--------|------|-------------|
| `"fixed"` | Low | Same size every trade |
| `"volatility"` | Medium | Size based on volatility |
| `"confidence"` | Medium | Size based on model confidence |
| `"kelly"` | High | Optimal growth (aggressive) |

---

## Complete Config Reference Table

| Variable | Type | Default | Options |
|----------|------|---------|---------|
| `SYMBOL` | str | "MES" | Any string |
| `DATA_PATH` | str | None | Valid file path |
| `MODELS` | list | ["xgboost", "lightgbm"] | See model list above |
| `OPTIMIZE_FOR` | str | "sharpe_ratio" | sharpe_ratio, sortino_ratio, profit_factor, f1_weighted, accuracy |
| `HORIZONS` | list | [5, 10, 15, 20] | Any positive integers |
| `BUILD_ENSEMBLE` | bool | True | True, False |
| `META_LEARNER` | str | "ridge_meta" | ridge_meta, mlp_meta, xgboost_meta, calibrated_meta |
| `OPTUNA_TRIALS` | int | 50 | Any positive integer |
| `LABELING_METHOD` | str | "triple_barrier" | triple_barrier, directional, threshold |
| `FEATURE_FAMILIES` | list | ["price", "momentum", "volatility", "volume", "trend"] | See feature list above |
| `RUN_BACKTEST` | bool | True | True, False |
| `POSITION_SIZING` | str | "fixed" | fixed, kelly, volatility, confidence |
| `EXPERIMENT_NAME` | str | "my_experiment" | Any string |
| `RANDOM_SEED` | int | 42 | Any integer |
| `SAVE_RESULTS` | bool | True | True, False |

---

## Tips and Best Practices

### Getting Started
1. **Start simple:** Use 2 boosting models + 1 horizon first
2. **Verify GPU:** Run `import torch; print(torch.cuda.is_available())`
3. **Save results:** Always enable `SAVE_TO_DRIVE = True`

### Model Selection
- **Always include:** XGBoost or LightGBM (fast, reliable baseline)
- **Add neural:** LSTM or GRU if you have GPU time
- **Transformers last:** PatchTST/iTransformer for production only

### Optimization
- **25 trials:** Quick tests
- **50 trials:** Development
- **100+ trials:** Production

### Features
- **Core set:** price, momentum, volatility (always include)
- **Add volume:** If your data has reliable volume
- **Advanced:** microstructure, regime (adds complexity)

---

## Troubleshooting

### "CUDA not available"
```python
import torch
print(torch.cuda.is_available())  # Should be True
```
**Fix:** Change runtime to GPU: Kernel → "Colab" → Select T4/A100

### "Module not found"
**Fix:** Re-run Cell 1 to install dependencies

### "Drive mount hangs" (VS Code Extension)
**Cause:** Drive mounting is not supported in VS Code extension
**Fix:** Upload data directly or use URL download (see Data Upload section)

### "Out of memory"
**Fix:** Reduce `MODELS` list or use fewer `HORIZONS`

### "Notebook disconnected"
**Cause:** Colab timeout after inactivity
**Fix:** Reconnect kernel and resume from checkpoint (if enabled)

---

## Python Version

This project requires **Python 3.11+** and is fully compatible with **Python 3.12.12**.

Google Colab currently runs Python 3.10 or 3.11 depending on runtime.

---

## Getting Help

- **Issues:** [GitHub Issues](https://github.com/Snehpatel101/Research/issues)
- **Documentation:** See `DIRECTION.md` for architecture details
- **Config Source:** `src/config/experiment.py`, `src/config/data.py`

---

*Last updated: 2026-01-28*
