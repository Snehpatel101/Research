# Inference and Contracts System

**Created:** 2026-01-23 (from Phase 1 Agent #6 analysis)
**Status:** Production-ready

---

## Overview

The inference system provides production-grade model serving with:
- Contract-based input/output validation
- Bundle packaging for deployment
- Backtesting framework integration
- PreprocessingGraph for train/serve parity

---

## Contract System

### DataContract

Validates data flowing through the pipeline:
- **Schema validation:** columns, dtypes, shape
- **DataRank:** 2D (tabular), 3D (sequence), 4D (multi-stream)
- **FeatureMode:** REDUCED, STANDARD, FULL
- **MTFMode:** SINGLE, MULTI, AGGREGATED

### ModelContract (23 Models)

Each model declares its input requirements:
- Required features
- Sequence length
- Scaler type
- MTF mode
- Data rank requirements

**Model Families:**
- Boosting (3): XGBoost, LightGBM, CatBoost
- Neural (10): LSTM, GRU, TCN, Transformer, PatchTST, iTransformer, TFT, N-BEATS, InceptionTime, ResNet1D
- Classical (3): Random Forest, Logistic, SVM
- Ensemble (3): Voting, Stacking, Blending
- Meta-learners (4): Ridge Meta, MLP Meta, Calibrated Meta, XGBoost Meta

### ArtifactManifest

Tracks reproducibility information:
- SHA256 hashes for config, data, artifact
- Environment capture: Python + package versions
- Lineage: pipeline_run_id, training_run_id

---

## Bundle System

Model bundles package everything needed for inference:

```
bundle_dir/
  manifest.json
  metadata.json
  features.json
  scaler.pkl
  calibrator.pkl (optional)
  preprocessing_graph.json (optional)
  model/
```

---

## PreprocessingGraph (Train/Serve Parity)

Captures all preprocessing transformations for reproducible inference:

- **CleaningConfig:** resample, gap fill, outliers
- **IndicatorConfig:** 20+ indicator periods
- **MTFConfig:** timeframes, mode
- **WaveletConfig:** decomposition settings
- **RegimeConfig:** detection thresholds
- **ScalingConfig:** robust/standard/minmax

**Purpose:** Ensures inference uses identical preprocessing to training.

---

## InferenceOrchestrator (PHASE_5 Entry Point)

Main API for model serving:

```python
from src.inference import InferenceOrchestrator

# Load from experiment
orch = InferenceOrchestrator.from_experiment(config)

# Prediction methods
orch.predict(X)                      # Single model
orch.predict_all(X)                  # All loaded models
orch.predict_from_raw(raw_df)        # With preprocessing
orch.predict_batch(data, batch_size) # Large datasets
orch.predict_with_uncertainty(X)     # Confidence estimates
```

---

## Backtesting Framework

Realistic backtesting with:

### BacktestConfig
- MES/MGC presets with realistic costs
- Configurable slippage models

### ExecutionModel
- MARKET_ON_CLOSE
- MARKET_ON_OPEN
- VWAP

### Position Sizing
- Fixed
- Kelly
- FixedFractional
- VolTargeted

### Metrics
- Sharpe, Sortino, Calmar
- Max Drawdown
- VAR, CVAR

---

## Known Issues

1. **Limited ensemble integration** - Ensemble bundles not fully supported
2. **No streaming batch option** - Batch inference requires full dataset
3. **Preprocessing graph complexity** - ~50 parameters may be overwhelming

---

## Phase 2 Recommendations

1. Add comprehensive input validation to bundles
2. Create preprocessing graph presets (minimal, standard, full)
3. Implement bundle caching for faster loading
4. Add data quality validators to contracts

---

**Last Updated:** 2026-01-23
