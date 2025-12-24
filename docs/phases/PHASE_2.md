# Phase 2 – Model Factory: Training Any Model Family

## Overview

Phase 2 builds a **modular model training factory** that supports any model family through a plugin architecture. Rather than hardcoding specific models, we create infrastructure that allows you to plug in any model (boosting, neural, time series, classical) and get comparable results.

**Phase 2 Goal:** Build the MODEL FACTORY infrastructure - a training system where adding a new model is just an interface + config, not a rewrite.

**Think of Phase 2 as:** Building a production line for training models, not building specific models.

---

## Core Concept: Plugin Architecture

### Models as Plugins, Not Hardcoded Implementations

Every model implements the same interface:

```python
class BaseModel(ABC):
    @abstractmethod
    def fit(self, X_train, y_train, X_val, y_val) -> Dict: ...
    @abstractmethod
    def predict(self, X, metadata) -> PredictionOutput: ...
    @abstractmethod
    def save(self, path: Path): ...
    @abstractmethod
    def load(self, path: Path): ...
```

Models register themselves:

```python
@ModelRegistry.register(name="xgboost", family="boosting")
class XGBoostModel(BaseModel):
    ...

# Auto-discovery + factory instantiation
model = ModelRegistry.create("xgboost", config, horizon, features)
```

### Supported Model Families

The factory supports ANY model family that implements the interface:

#### Boosting Models (Fast, Interpretable)
- **XGBoost** - Industry standard gradient boosting
- **LightGBM** - Microsoft's fast boosting library
- **CatBoost** - Yandex's categorical-aware boosting

#### Time Series Specialists (Multi-Horizon Experts)
- **TCN** - Temporal Convolutional Networks (dilated causal convolutions, parallelizable)
- **N-HiTS** - MLP-based multi-scale forecaster
- **TFT** - Temporal Fusion Transformer (attention-based, interpretable)
- **PatchTST** - Patch-based transformer (efficient long-range)
- **Informer** - Sparse attention for long sequences (AAAI 2021)
- **Autoformer** - Auto-correlation decomposition (NeurIPS 2021, 38% better than Informer)
- **TimesNet** - 2D variation modeling (ICLR 2023, SOTA classification)
- **WaveNet** - Dilated causal convolutions (generative, probabilistic)

#### Foundation Models (Pre-trained, Zero-Shot Capable)
- **TimesFM 2.0** - Google's 500M param foundation model (2048 context length)
- **TimeLLM** - LLM reprogramming for time series (ICLR 2024)

#### Neural Networks (Sequential Learning)
- **LSTM** - Long Short-Term Memory networks
- **GRU** - Gated Recurrent Units
- **Transformer** - Self-attention architecture
- **RWKV-TS** - Efficient RNN alternative (2024)

#### Classical Baselines (Benchmarks)
- **RandomForest** - Ensemble of decision trees
- **SVM** - Support Vector Machines
- **LogisticRegression** - Simple linear baseline

**Adding a new model requires:**
1. Implement the 4 methods (fit, predict, save, load)
2. Register with decorator: `@ModelRegistry.register(name, family)`
3. Add YAML config file
4. Done - no changes to training infrastructure

---

## Objectives

### Primary Goals
- Build BaseModel interface and ModelRegistry
- Build TimeSeriesDataset with zero-leakage windowing
- Build Trainer orchestration (data → model → train → evaluate)
- Build ModelEvaluator (metrics + backtests)
- Implement 3-5 example models from different families
- Generate out-of-sample predictions on validation set
- Log comprehensive metrics (accuracy, F1, Sharpe, drawdown)

### Success Criteria
- Registry can discover and instantiate any registered model
- TimeSeriesDataset correctly windows sequences with symbol isolation
- Trainer works with ANY model via interface
- Each model achieves validation F1 > 0.35 on at least one horizon
- Each model achieves validation Sharpe > 0.3 on at least one horizon
- Training completes within reasonable time (<24 hours per model on single GPU)
- All models generate consistent probability outputs for ensemble stacking

---

## Prerequisites

### Required Inputs from Phase 1
- `data/splits/scaled/train_scaled.parquet` (87,094 rows × 126 cols)
- `data/splits/scaled/val_scaled.parquet` (18,591 rows × 126 cols)
- `data/splits/scaled/test_scaled.parquet` (18,592 rows × 126 cols)
- `data/splits/scaled/feature_scaler.json` (scaling metadata)
- `config/ga_results/` (GA-optimized barrier parameters for reference)

### Infrastructure Requirements
- **GPU:** 1× RTX 3090/4090/5090 or A100/H100 (16GB+ VRAM for neural models)
- **RAM:** 32-64 GB
- **Storage:** 50-100 GB for checkpoints and logs
- **Compute Time:** ~2-24 hours per model (boosting: 2h, transformers: 8-24h)

### Software Dependencies
```bash
# Core ML
pip install xgboost lightgbm catboost
pip install scikit-learn>=1.3

# Neural Networks
pip install torch>=2.0
pip install lightning>=2.0

# Time Series Libraries (SOTA Models)
pip install neuralforecast>=1.6      # N-HiTS, TCN, PatchTST, TimesNet
pip install pytorch-forecasting>=1.0 # TFT
pip install darts>=0.27              # TCN, WaveNet, Informer (alternative)
pip install transformers>=4.35       # PatchTST (HuggingFace)
pip install timesfm                  # Google TimesFM foundation model

# For Autoformer, TimesNet, Informer (official implementations)
# Clone: github.com/thuml/Time-Series-Library

# Experiment Tracking
pip install mlflow>=2.0
pip install optuna>=3.0  # Hyperparameter tuning

# Utilities
pip install numpy>=1.24 pandas>=2.0
```

---

## Architecture: The Model Factory

### 1. BaseModel (Abstract Interface)

**File:** `src/models/base.py` (~250 lines)

```python
@dataclass
class ModelConfig:
    model_name: str
    model_family: str
    horizon: int  # 5 or 20
    random_seed: int = 42
    early_stopping: bool = True
    patience: int = 10

@dataclass
class PredictionOutput:
    predictions: np.ndarray        # Shape: (n,), values: {-1, 0, 1}
    probabilities: np.ndarray      # Shape: (n, 3)
    timestamps: pd.DatetimeIndex
    symbols: np.ndarray
    horizons: np.ndarray

class BaseModel(ABC):
    @abstractmethod
    def fit(self, X_train, y_train, X_val, y_val, metadata_train, metadata_val) -> Dict: ...
    @abstractmethod
    def predict(self, X, metadata) -> PredictionOutput: ...
    @abstractmethod
    def save(self, path: Path): ...
    @abstractmethod
    def load(self, path: Path): ...
```

### 2. ModelRegistry (Plugin System)

**File:** `src/models/registry.py` (~180 lines)

```python
class ModelRegistry:
    _registry: Dict[str, Type[BaseModel]] = {}

    @classmethod
    def register(cls, name: str, family: str):
        """Decorator to register a model class."""
        def decorator(model_class: Type[BaseModel]):
            # Validation
            if not issubclass(model_class, BaseModel):
                raise TypeError(f"Model must inherit from BaseModel")

            # Register
            full_name = f"{family}:{name}"
            cls._registry[full_name] = model_class
            return model_class
        return decorator

    @classmethod
    def create(cls, model_name: str, config: dict, horizon: int, feature_columns: List[str]) -> BaseModel:
        """Factory method to create a model."""
        full_name = cls._resolve_name(model_name)
        if full_name not in cls._registry:
            raise ValueError(f"Model '{model_name}' not found")

        model_class = cls._registry[full_name]
        return model_class(config=config, horizon=horizon, feature_columns=feature_columns)

    @classmethod
    def list_models(cls, family: Optional[str] = None) -> List[dict]:
        """List all registered models."""
        ...
```

### 3. TimeSeriesDataset (Temporal Windowing)

**File:** `src/data/dataset.py` (~200 lines)

```python
class TimeSeriesDataset:
    """
    Creates temporal sequences with zero-leakage guarantees.

    Key Features:
    - Symbol-isolated windows (no cross-symbol leakage)
    - Past features only (no future leakage)
    - Flexible sequence lengths (1 for boosting, 60+ for time series)
    - Purge/embargo already applied by Phase 1
    """

    def __init__(self, config: DatasetConfig):
        self.seq_len = config.sequence_length
        self.horizon = config.horizon

        # Load Phase 1 splits
        self.train_df = pd.read_parquet(config.train_path)
        self.val_df = pd.read_parquet(config.val_path)

        # Auto-detect features vs labels
        self.feature_cols = [c for c in self.train_df.columns
                           if c not in ['datetime', 'symbol', 'split']
                           and not c.startswith('label_')]

        self.label_col = f'label_h{self.horizon}'

    def _create_sequences(self, df: pd.DataFrame, split: str):
        """Create sequences per symbol."""
        sequences = []

        for symbol in df['symbol'].unique():
            symbol_df = df[df['symbol'] == symbol].reset_index(drop=True)

            for i in range(self.seq_len, len(symbol_df)):
                # Past features only
                X_window = symbol_df.iloc[i-self.seq_len:i][self.feature_cols].values

                # Future label (at step i)
                y = symbol_df.iloc[i][self.label_col]

                # Metadata
                meta = {
                    'datetime': symbol_df.iloc[i]['datetime'],
                    'symbol': symbol,
                    'horizon': self.horizon
                }

                sequences.append((X_window, y, meta))

        return sequences
```

### 4. Trainer (Orchestration)

**File:** `src/training/trainer.py` (~200 lines)

```python
class Trainer:
    """
    Orchestrates the full training workflow.

    Delegates actual training to model.fit() but handles:
    - Data loading
    - Model instantiation via registry
    - Metric logging
    - Artifact management
    - MLflow tracking
    """

    def __init__(self, config: TrainerConfig):
        self.config = config
        self.model = None
        self.dataset = None

    def run_full_pipeline(self):
        """End-to-end training pipeline."""
        self.prepare_data()      # Load Phase 1 splits
        self.build_model()       # Instantiate via registry
        self.train()             # Delegate to model.fit()
        self.evaluate()          # Compute metrics, save predictions

    def prepare_data(self):
        self.dataset = TimeSeriesDataset(self.config.dataset_config)

    def build_model(self):
        self.model = ModelRegistry.create(
            model_name=self.config.model_name,
            config=self.config.model_config,
            horizon=self.config.horizon,
            feature_columns=self.dataset.feature_cols
        )

    def train(self):
        # Get data
        X_train, y_train, meta_train = self.dataset.get_split('train')
        X_val, y_val, meta_val = self.dataset.get_split('val')

        # Delegate to model
        self.history = self.model.fit(
            X_train, y_train, X_val, y_val,
            meta_train, meta_val
        )

    def evaluate(self):
        # Get predictions
        X_val, y_val, meta_val = self.dataset.get_split('val')
        preds = self.model.predict(X_val, meta_val)

        # Compute metrics
        evaluator = ModelEvaluator()
        metrics = evaluator.compute_all_metrics(preds, y_val)

        # Save
        self.save_predictions(preds)
        self.save_metrics(metrics)
```

### 5. ModelEvaluator (Metrics)

**File:** `src/training/evaluator.py` (~150 lines)

```python
class ModelEvaluator:
    """Computes classification and trading metrics."""

    def compute_all_metrics(self, predictions: PredictionOutput, y_true: np.ndarray) -> Dict:
        metrics = {}

        # Classification metrics
        metrics['accuracy'] = accuracy_score(y_true, predictions.predictions)
        metrics['f1_macro'] = f1_score(y_true, predictions.predictions, average='macro')
        metrics['f1_per_class'] = f1_score(y_true, predictions.predictions, average=None)

        # Trading metrics (simulate simple strategy)
        sharpe, max_dd, win_rate = self.backtest_predictions(predictions)
        metrics['sharpe'] = sharpe
        metrics['max_drawdown'] = max_dd
        metrics['win_rate'] = win_rate

        return metrics

    def backtest_predictions(self, predictions: PredictionOutput) -> Tuple[float, float, float]:
        """Simple rule-based backtest."""
        # Convert predictions to signals: {-1, 0, 1}
        signals = predictions.predictions

        # Simulate returns (simplified)
        # ... use vectorbt or custom logic

        return sharpe, max_dd, win_rate
```

---

## Example Model Configurations

### Boosting Example: XGBoost

**File:** `src/models/boosting/xgboost.py` (~180 lines)

```python
@dataclass
class XGBoostConfig(ModelConfig):
    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8

@ModelRegistry.register(name="xgboost", family="boosting")
class XGBoostModel(BaseModel):
    def _build_config(self, config: dict, horizon: int) -> XGBoostConfig:
        return XGBoostConfig(
            model_name='xgboost',
            model_family='boosting',
            horizon=horizon,
            **config
        )

    def _build_model(self):
        self.model = xgb.XGBClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            objective='multi:softprob',
            num_class=3,
            random_state=self.config.random_seed
        )

    def fit(self, X_train, y_train, X_val, y_val, metadata_train, metadata_val) -> Dict:
        # Validate
        self.validate_inputs(X_train, y_train)

        # Flatten 3D if needed (boosting expects 2D)
        if X_train.ndim == 3:
            X_train = X_train.reshape(len(X_train), -1)
            X_val = X_val.reshape(len(X_val), -1)

        # Encode labels: {-1, 0, 1} → {0, 1, 2}
        y_train_enc = y_train + 1
        y_val_enc = y_val + 1

        # Train with early stopping
        self.model.fit(
            X_train, y_train_enc,
            eval_set=[(X_val, y_val_enc)],
            verbose=False
        )

        self.is_fitted = True
        return {'best_iteration': self.model.best_iteration}

    def predict(self, X, metadata) -> PredictionOutput:
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")

        # Flatten 3D
        if X.ndim == 3:
            X = X.reshape(len(X), -1)

        # Predict probabilities
        proba = self.model.predict_proba(X)
        pred_enc = self.model.predict(X)

        # Decode: {0, 1, 2} → {-1, 0, 1}
        predictions = pred_enc - 1

        return PredictionOutput(
            predictions=predictions,
            probabilities=proba,
            timestamps=pd.to_datetime(metadata['datetime']),
            symbols=metadata['symbol'].values,
            horizons=np.array([self.horizon] * len(X))
        )

    def save(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(path / 'xgboost_model.json'))
        with open(path / 'metadata.pkl', 'wb') as f:
            pickle.dump({
                'config': self.config,
                'feature_columns': self.feature_columns
            }, f)

    def load(self, path: Path):
        self.model.load_model(str(path / 'xgboost_model.json'))
        with open(path / 'metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
        self.config = metadata['config']
        self.feature_columns = metadata['feature_columns']
        self.is_fitted = True
```

**Config:** `config/models/xgboost.yaml`

```yaml
model_name: xgboost
model_family: boosting
n_estimators: 100
max_depth: 6
learning_rate: 0.1
subsample: 0.8
colsample_bytree: 0.8
early_stopping: true
patience: 10
```

### Time Series Example: N-HiTS

**File:** `src/models/timeseries/nhits.py` (~220 lines)

```python
@ModelRegistry.register(name="nhits", family="timeseries")
class NHiTSModel(BaseModel):
    """N-HiTS with multi-horizon classification heads."""

    def _build_model(self):
        # Use NeuralForecast's NHiTS as encoder
        from neuralforecast.models import NHITS

        self.encoder = NHITS(
            h=self.config.output_size,
            input_size=self.config.input_size,
            n_blocks=self.config.n_blocks,
            mlp_units=self.config.mlp_units,
            dropout_prob=self.config.dropout
        )

        # Add classification heads
        H_mid = 256
        self.shared_mlp = nn.Sequential(
            nn.Linear(512, H_mid),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.classification_head = nn.Linear(H_mid, 3)

    def fit(self, X_train, y_train, X_val, y_val, metadata_train, metadata_val) -> Dict:
        # PyTorch training loop
        # ... similar structure to XGBoost but with neural training

        return {'train_loss': train_losses, 'val_loss': val_losses}
```

**Config:** `config/models/nhits.yaml`

```yaml
model_name: nhits
model_family: timeseries
input_size: 128
output_size: 20
n_blocks: [1, 1, 1]
mlp_units: [[512, 512], [512, 512], [512, 512]]
dropout: 0.2
learning_rate: 0.001
batch_size: 64
max_epochs: 100
```

---

## Usage: Training Any Model

### CLI Script

**File:** `scripts/train_model.py`

```bash
#!/usr/bin/env python
"""Train any registered model."""

python scripts/train_model.py \
    --model xgboost \
    --horizon 5 \
    --config config/models/xgboost.yaml \
    --output experiments/runs/xgboost_h5_20251224

python scripts/train_model.py \
    --model nhits \
    --horizon 20 \
    --config config/models/nhits.yaml \
    --output experiments/runs/nhits_h20_20251224

python scripts/train_model.py \
    --model lightgbm \
    --horizon 5 \
    --config config/models/lightgbm.yaml \
    --output experiments/runs/lightgbm_h5_20251224
```

### Programmatic Usage

```python
from src.models.registry import ModelRegistry
from src.data.dataset import TimeSeriesDataset
from src.training.trainer import Trainer

# List available models
models = ModelRegistry.list_models()
print(f"Available: {[m['name'] for m in models]}")

# Create and train any model
model = ModelRegistry.create(
    model_name='xgboost',
    config={'n_estimators': 50, 'max_depth': 4},
    horizon=5,
    feature_columns=feature_cols
)

# Train
trainer = Trainer(config=trainer_config)
trainer.model = model
trainer.run_full_pipeline()
```

---

## Deliverables

### Core Infrastructure (Week 1-2)
- ✅ `src/models/base.py` - BaseModel, ModelConfig, PredictionOutput
- ✅ `src/models/registry.py` - ModelRegistry
- ✅ `src/data/dataset.py` - TimeSeriesDataset
- ✅ `src/training/trainer.py` - Trainer orchestration
- ✅ `src/training/evaluator.py` - ModelEvaluator
- ✅ Unit tests for all components

### Example Models (Week 3-4)
- ✅ `src/models/boosting/xgboost.py` - XGBoost implementation
- ✅ `src/models/boosting/lightgbm.py` - LightGBM implementation
- ✅ `src/models/boosting/catboost.py` - CatBoost implementation
- ✅ `src/models/timeseries/nhits.py` - N-HiTS implementation
- ✅ `src/models/timeseries/tft.py` - TFT implementation
- ✅ Config YAML files for all models

### Artifacts Per Model
For each trained model:
1. **Model Weights:** `experiments/runs/{model}_{timestamp}/checkpoints/`
2. **Predictions:** `experiments/runs/{model}_{timestamp}/predictions/val_predictions.parquet`
3. **Metrics:** `experiments/runs/{model}_{timestamp}/metrics/metrics.json`
4. **Plots:** `experiments/runs/{model}_{timestamp}/plots/`

### Comparative Analysis
- `experiments/model_comparison.md` - Side-by-side metrics for all models
- Best model per horizon
- Prediction correlations (diversity analysis)
- Training time comparisons

---

## Success Criteria

Phase 2 is complete when:

1. ✅ **Factory infrastructure works:** ModelRegistry can discover and instantiate any registered model
2. ✅ **TimeSeriesDataset validated:** Correct windowing, no leakage, symbol isolation
3. ✅ **Trainer generalized:** Works with ANY model via BaseModel interface
4. ✅ **3+ model families implemented:** At least one from boosting, time series, and classical
5. ✅ **All models meet thresholds:** F1 > 0.35 AND Sharpe > 0.3 on at least one horizon
6. ✅ **Consistent outputs:** All models generate same format predictions for ensemble
7. ✅ **Documentation complete:** Usage guides, API docs, example configs
8. ✅ **Tests passing:** >80% coverage on core infrastructure

**Proceed to Phase 3** (Ensemble Stacking) only after infrastructure is proven with multiple model families.

---

## Implementation Checklist

### Week 1: Core Infrastructure
- [ ] Implement BaseModel abstract class
- [ ] Implement ModelRegistry with decorator registration
- [ ] Implement TimeSeriesDataset with windowing
- [ ] Write unit tests for validation logic
- [ ] Test end-to-end with dummy data

### Week 2: First Model Family (Boosting)
- [ ] Implement XGBoostModel
- [ ] Implement LightGBMModel
- [ ] Implement CatBoostModel
- [ ] Test with real Phase 1 data
- [ ] Verify save/load roundtrip

### Week 3: Training Infrastructure
- [ ] Implement Trainer orchestration
- [ ] Implement ModelEvaluator
- [ ] Implement training callbacks (early stopping, checkpointing)
- [ ] Create CLI scripts (train_model.py, run_experiment.py)
- [ ] Add MLflow integration

### Week 4: Time Series Models
- [ ] Implement N-HiTS model
- [ ] Implement TFT model (optional: PatchTST)
- [ ] Run baseline experiments (all models, both horizons)
- [ ] Generate comparative analysis report

### Week 5: Hyperparameter Tuning (Optional)
- [ ] Implement OptunaModelTuner
- [ ] Define search spaces per model family
- [ ] Run tuning experiments (50-100 trials per model)
- [ ] Lock in production configs

---

## Notes

### Adding a New Model

To add a new model to the factory:

1. **Create model file:** `src/models/{family}/{model_name}.py`
2. **Implement interface:**
   ```python
   @ModelRegistry.register(name="mymodel", family="myfamily")
   class MyModel(BaseModel):
       def fit(...): ...
       def predict(...): ...
       def save(...): ...
       def load(...): ...
   ```
3. **Add config:** `config/models/mymodel.yaml`
4. **Test:** Import and verify it appears in `ModelRegistry.list_models()`

### Why Plugin Architecture?

- **Extensibility:** Add models without touching core code
- **Comparability:** Same metrics, same evaluation, fair comparison
- **Maintainability:** Each model isolated, changes don't cascade
- **Testing:** Test infrastructure once, not per-model

### Model Family Characteristics

| Family | Strengths | Weaknesses | Training Time |
|--------|-----------|------------|---------------|
| Boosting | Fast, interpretable, strong baseline | No sequence modeling | 1-3 hours |
| Time Series | Multi-horizon, sequential patterns | Slow, needs GPU, harder to tune | 8-24 hours |
| Neural | Flexible, can learn complex patterns | Needs lots of data, prone to overfit | 4-12 hours |
| Classical | Simple, fast, good baseline | Limited capacity | <1 hour |

---

**End of Phase 2 Specification: Model Factory Infrastructure**
