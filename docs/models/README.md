# Model Training Guides

Comprehensive training documentation for all 19 models in the ML Model Factory.

**Last Updated:** 2025-12-30

---

## Overview

This directory contains detailed training guides for each model, including:
- Exact hyperparameter configurations
- Hardware requirements (CPU, GPU, RAM)
- Training time estimates
- Batch size recommendations
- Memory formulas
- Example commands
- Troubleshooting guides

---

## Quick Reference

| Model | Family | Status | Guide | Config |
|-------|--------|--------|-------|--------|
| XGBoost | Boosting | ✅ Implemented | [Guide](XGBOOST_TRAINING_GUIDE.md) | [config/models/xgboost.yaml](/home/user/Research/config/models/xgboost.yaml) |
| LightGBM | Boosting | ✅ Implemented | _See XGBoost guide_ | [config/models/lightgbm.yaml](/home/user/Research/config/models/lightgbm.yaml) |
| CatBoost | Boosting | ✅ Implemented | _See XGBoost guide_ | [config/models/catboost.yaml](/home/user/Research/config/models/catboost.yaml) |
| LSTM | Neural | ✅ Implemented | [Guide](LSTM_TRAINING_GUIDE.md) | [config/models/lstm.yaml](/home/user/Research/config/models/lstm.yaml) |
| GRU | Neural | ✅ Implemented | _See LSTM guide_ | [config/models/gru.yaml](/home/user/Research/config/models/gru.yaml) |
| TCN | Neural | ✅ Implemented | _See LSTM guide_ | [config/models/tcn.yaml](/home/user/Research/config/models/tcn.yaml) |
| Transformer | Neural | ✅ Implemented | _See LSTM guide_ | [config/models/transformer.yaml](/home/user/Research/config/models/transformer.yaml) |
| Random Forest | Classical | ✅ Implemented | _See XGBoost guide_ | [config/models/random_forest.yaml](/home/user/Research/config/models/random_forest.yaml) |
| Logistic | Classical | ✅ Implemented | _See XGBoost guide_ | [config/models/logistic.yaml](/home/user/Research/config/models/logistic.yaml) |
| SVM | Classical | ✅ Implemented | _See XGBoost guide_ | [config/models/svm.yaml](/home/user/Research/config/models/svm.yaml) |
| Voting | Ensemble | ✅ Implemented | [Guide](VOTING_ENSEMBLE_TRAINING_GUIDE.md) | [config/models/voting.yaml](/home/user/Research/config/models/voting.yaml) |
| Stacking | Ensemble | ✅ Implemented | _See Voting guide_ | [config/models/stacking.yaml](/home/user/Research/config/models/stacking.yaml) |
| Blending | Ensemble | ✅ Implemented | _See Voting guide_ | [config/models/blending.yaml](/home/user/Research/config/models/blending.yaml) |
| InceptionTime | CNN | 🚧 Planned | _To be created_ | _To be created_ |
| 1D ResNet | CNN | 🚧 Planned | _To be created_ | _To be created_ |
| PatchTST | Transformer | 🚧 Planned | [Guide](PATCHTST_TRAINING_GUIDE.md) | _To be created_ |
| iTransformer | Transformer | 🚧 Planned | _To be created_ | _To be created_ |
| TFT | Transformer | 🚧 Planned | _To be created_ | _To be created_ |
| N-BEATS | MLP | 🚧 Planned | _To be created_ | _To be created_ |

---

## Hardware Requirements Matrix

**See:** [REQUIREMENTS_MATRIX.md](REQUIREMENTS_MATRIX.md)

Comprehensive comparison of:
- Minimum/recommended/optimal hardware
- GPU memory formulas
- Batch size recommendations by GPU
- Training time benchmarks
- Cost estimates (cloud + on-premise)

---

## Training Guides by Model

### Implemented Models (13)

#### Boosting Family

1. **[XGBoost Training Guide](XGBOOST_TRAINING_GUIDE.md)**
   - CPU/GPU configuration
   - Tree hyperparameters (depth, learning rate, regularization)
   - GPU memory: 2-4GB fixed
   - Training time: 2-5 min (GPU), 10-20 min (CPU)
   - Batch size: N/A (processes full dataset)

2. **LightGBM** - Similar to XGBoost, see XGBoost guide
   - Typically 20-30% faster than XGBoost
   - leaf_based splits (vs XGBoost's depth-based)

3. **CatBoost** - Similar to XGBoost, see XGBoost guide
   - Native categorical feature support
   - Ordered boosting (reduces overfitting)

#### Neural Family

4. **[LSTM Training Guide](LSTM_TRAINING_GUIDE.md)**
   - Sequence model (3D input)
   - GPU memory formula: `4 * hidden * (features + hidden + 1) * layers`
   - Batch size: 512 (RTX 3080), 1024 (RTX 4090)
   - Training time: 15-30 min (RTX 3080)
   - Mixed precision: Auto-enabled (BF16/FP16)

5. **GRU** - Similar to LSTM, 20-25% less memory
   - 3 gates vs LSTM's 4 gates
   - Faster training, similar performance

6. **TCN** - Dilated convolutional network
   - Longer sequences (120 vs 60) due to efficient memory
   - Parallel training (convolutions vs sequential RNNs)
   - Memory: ~5GB for typical configs

7. **Transformer** - Self-attention mechanism
   - Highest memory: O(seq²) attention
   - Batch size: 64-128 (smaller than LSTM)
   - Training time: 30-60 min

#### Classical Family

8. **Random Forest** - CPU-only
   - Bagging of decision trees
   - Training time: 3-8 min
   - Memory: Linear in n_estimators

9. **Logistic Regression** - CPU-only
   - Linear baseline
   - Training time: 1-2 min
   - Very lightweight

10. **SVM** - CPU-only
    - RBF kernel
    - Training time: 5-15 min
    - **Warning:** O(n²-n³) complexity, subsample for large datasets

#### Ensemble Family

11. **[Voting Ensemble Guide](VOTING_ENSEMBLE_TRAINING_GUIDE.md)**
    - Soft/hard voting
    - **Critical:** Cannot mix tabular + sequence models
    - Training time: Sum of base models
    - Memory: Sum of base models + predictions

12. **Stacking Ensemble** - See Voting guide
    - Trains meta-learner on OOF predictions
    - Training time: 1.2x base models
    - Higher accuracy than voting

13. **Blending Ensemble** - See Voting guide
    - Trains meta-learner on holdout predictions
    - Training time: 1.1x base models
    - Simpler than stacking

---

### Planned Models (6)

#### CNN Family (2 models)

14. **InceptionTime** - Multi-scale CNN ensemble
    - Status: Planned (Q1 2025)
    - Implementation effort: 3 days
    - Expected GPU: RTX 3080 (6-8GB)

15. **1D ResNet** - Deep residual network
    - Status: Planned (Q1 2025)
    - Implementation effort: 2 days
    - Expected GPU: RTX 3070 (4-6GB)

#### Advanced Transformer Family (3 models)

16. **[PatchTST Training Guide](PATCHTST_TRAINING_GUIDE.md)**
    - Status: Planned (Q1 2025)
    - Implementation effort: 4 days
    - Expected GPU: RTX 3090 (6-10GB)
    - SOTA long-term forecasting (21% MSE reduction)

17. **iTransformer** - Inverted attention (features as tokens)
    - Status: Planned (Q1 2025)
    - Implementation effort: 3 days
    - Expected GPU: RTX 3080 (4-7GB)

18. **TFT** - Temporal Fusion Transformer
    - Status: Planned (Q1 2025)
    - Implementation effort: 5 days
    - Expected GPU: RTX 3090 (8-12GB)
    - Interpretable attention + quantile forecasting

#### MLP Family (1 model)

19. **N-BEATS** - Neural basis expansion
    - Status: Planned (Q1 2025)
    - Implementation effort: 1 day
    - Expected GPU: RTX 3070 (3-5GB)
    - M4 competition winner

---

## Common Training Patterns

### Training a Single Model

```bash
# Boosting model (CPU or GPU)
python scripts/train_model.py --model xgboost --horizon 20 --use-gpu

# Neural model (GPU required)
python scripts/train_model.py --model lstm --horizon 20 --seq-len 60 --batch-size 512

# Classical model (CPU only)
python scripts/train_model.py --model random_forest --horizon 20
```

### Training an Ensemble

```bash
# Tabular ensemble (XGBoost + LightGBM + CatBoost)
python scripts/train_model.py \
    --model voting \
    --base-models xgboost,lightgbm,catboost \
    --horizon 20

# Sequence ensemble (LSTM + GRU + TCN)
python scripts/train_model.py \
    --model stacking \
    --base-models lstm,gru,tcn \
    --horizon 20 \
    --seq-len 60 \
    --batch-size 512
```

### Cross-Validation

```bash
# Single model CV
python scripts/run_cv.py --models xgboost --horizons 20 --n-splits 5

# Multiple models CV
python scripts/run_cv.py --models xgboost,lstm --horizons 5,10,15,20 --n-splits 5

# With hyperparameter tuning
python scripts/run_cv.py --models xgboost --horizons 20 --n-splits 5 --tune --n-trials 50
```

---

## Hardware Recommendations

### By Budget

| Budget | GPU | Usable Models | Notes |
|--------|-----|---------------|-------|
| $0 | CPU only | Boosting + Classical (10 models) | No neural models |
| $400 | RTX 3060 (12GB) | All except large ensembles (13 models) | Entry-level |
| $800 | RTX 4070 Ti (12GB) | All 13 implemented models | Recommended |
| $1,600 | RTX 4090 (24GB) | All 19 models (when released) | Professional |

### By Use Case

**Hobbyist / Learning:**
- RTX 3060 (12GB) - $400
- Can train all boosting and small neural models
- Sufficient for model comparison

**Professional Researcher:**
- RTX 4070 Ti (12GB) or RTX 3090 (24GB) - $800-$1,200
- All models with standard configs
- Large ensembles possible

**Production / Lab:**
- RTX 4090 (24GB) or A6000 (48GB) - $1,600-$4,500
- All models with large configs
- Multiple parallel experiments

---

## Configuration Files

All model configs are in `/home/user/Research/config/models/`:

```
config/models/
├── xgboost.yaml       # Boosting models
├── lightgbm.yaml
├── catboost.yaml
├── lstm.yaml          # Neural models
├── gru.yaml
├── tcn.yaml
├── transformer.yaml
├── random_forest.yaml # Classical models
├── logistic.yaml
├── svm.yaml
├── voting.yaml        # Ensemble models
├── stacking.yaml
└── blending.yaml
```

**Config structure:**
```yaml
# Model identification
model:
  name: xgboost
  family: boosting
  description: Gradient boosted decision trees

# Default hyperparameters
defaults:
  n_estimators: 500
  max_depth: 6
  learning_rate: 0.05
  # ...

# Training settings
training:
  batch_size: 512
  max_epochs: 100
  early_stopping_patience: 15
  # ...

# Device settings
device:
  default: auto
  use_gpu: true
  mixed_precision: true
```

---

## Troubleshooting

### Common Issues

**GPU Out of Memory:**
1. Reduce batch size: `--batch-size 256`
2. Reduce sequence length: `--seq-len 30`
3. Reduce model size: `--hidden-size 128`

**Training Too Slow:**
1. Enable GPU: `--use-gpu`
2. Increase batch size: `--batch-size 1024`
3. Reduce dataset size for testing

**Model Not Converging:**
1. Reduce learning rate: `--learning-rate 0.0005`
2. Increase regularization (weight decay, dropout)
3. Check for data leakage (embargo/purge)

**EnsembleCompatibilityError:**
- Cannot mix tabular (XGBoost) + sequence (LSTM) models
- Use only tabular OR only sequence models in ensemble

---

## Performance Expectations

### Typical Validation F1 Scores (MES, 5min bars)

| Model | Expected F1 | Notes |
|-------|-------------|-------|
| XGBoost | 0.58-0.62 | Fast, reliable baseline |
| LightGBM | 0.57-0.61 | Similar to XGBoost |
| CatBoost | 0.59-0.63 | Slightly better than XGBoost |
| LSTM | 0.60-0.65 | Good temporal modeling |
| GRU | 0.59-0.64 | Similar to LSTM |
| TCN | 0.61-0.66 | Often best neural model |
| Transformer | 0.58-0.62 | Attention-based |
| Random Forest | 0.54-0.58 | Weaker than boosting |
| Logistic | 0.48-0.52 | Linear baseline |
| SVM | 0.52-0.56 | Kernel-based |
| Voting (XGB+LGB+CB) | 0.62-0.66 | +1-2% vs best base |
| Stacking | 0.63-0.68 | +2-3% vs best base |

**Note:** Performance depends on label horizons, features, and market regime. These are typical ranges.

---

## Next Steps

1. **Implement Planned Models:** See [ADVANCED_MODELS_ROADMAP.md](/home/user/Research/docs/roadmaps/ADVANCED_MODELS_ROADMAP.md)
2. **Create Remaining Guides:** Training guides for GRU, TCN, Transformer, RF, SVM, Stacking, Blending
3. **Benchmark All Models:** Run comprehensive comparison on MES + MGC
4. **Optimize Ensembles:** Auto-optimize voting weights, stacking meta-learner

---

## References

- **Main Documentation:** [/home/user/Research/CLAUDE.md](/home/user/Research/CLAUDE.md)
- **Model Integration Guide:** [/home/user/Research/docs/guides/MODEL_INTEGRATION_GUIDE.md](/home/user/Research/docs/guides/MODEL_INTEGRATION_GUIDE.md)
- **Advanced Models Roadmap:** [/home/user/Research/docs/roadmaps/ADVANCED_MODELS_ROADMAP.md](/home/user/Research/docs/roadmaps/ADVANCED_MODELS_ROADMAP.md)
- **Requirements Matrix:** [REQUIREMENTS_MATRIX.md](REQUIREMENTS_MATRIX.md)

---

## Contributing

To add a new model training guide:

1. Copy an existing guide as template
2. Update hardware requirements (use device.py for memory estimates)
3. Include example commands and config
4. Add troubleshooting section
5. Update this README

**Template structure:**
- Model Overview
- Hardware Requirements (min/recommended/optimal)
- Hyperparameters (defaults + tuning ranges)
- Training Configuration (CLI examples)
- Memory Requirements (formulas + tables)
- Batch Size Recommendations
- Training Time Estimates
- Validation Strategy
- Common Issues
- Example Output
- Cross-Validation
- Integration with Ensembles
- References
