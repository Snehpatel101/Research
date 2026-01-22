#!/usr/bin/env python
"""
ML for Dummies - Quick Start Demo

This script demonstrates the dead-simple API for training ML models
on OHLCV time series data.

The philosophy: Users make HIGH-LEVEL choices, the system figures out the details.
"""

# =============================================================================
# QUICK START - Just run these examples!
# =============================================================================

print("=" * 60)
print("ML FOR DUMMIES - Quick Start Demo")
print("=" * 60)

# Import the simple API
from src.config import (  # noqa: E402
    describe_model,
    list_models,
    preview_config,
    quick_compare,
    show_defaults,
)

# -----------------------------------------------------------------------------
# 1. LIST AVAILABLE MODELS
# -----------------------------------------------------------------------------
print("\n1. LIST AVAILABLE MODELS")
print("-" * 40)

# List all models
print("All models:", list_models())

# List models by family
print("\nBoosting models:", list_models("boosting"))
print("Neural models:", list_models("neural"))
print("Transformer models:", list_models("transformer"))

# -----------------------------------------------------------------------------
# 2. GET MODEL DESCRIPTIONS
# -----------------------------------------------------------------------------
print("\n\n2. GET MODEL DESCRIPTIONS")
print("-" * 40)

print(f"XGBoost: {describe_model('xgboost')}")
print(f"LSTM: {describe_model('lstm')}")
print(f"PatchTST: {describe_model('patchtst')}")

# -----------------------------------------------------------------------------
# 3. SEE SMART DEFAULTS
# -----------------------------------------------------------------------------
print("\n\n3. SEE SMART DEFAULTS")
print("-" * 40)

print("\nXGBoost defaults:")
for key, value in show_defaults("xgboost").items():
    print(f"  {key}: {value}")

print("\nLSTM defaults:")
for key, value in show_defaults("lstm").items():
    print(f"  {key}: {value}")

# -----------------------------------------------------------------------------
# 4. QUICK COMPARISON TABLE
# -----------------------------------------------------------------------------
print("\n\n4. QUICK COMPARISON TABLE")
print("-" * 40)

quick_compare()

# -----------------------------------------------------------------------------
# 5. PREVIEW CONFIGURATION (without training)
# -----------------------------------------------------------------------------
print("\n\n5. PREVIEW CONFIGURATION")
print("-" * 40)

# Preview what will happen with different settings
print("\nBasic LSTM:")
preview_config("lstm")

print("\nLSTM with optimization and custom sequence length:")
preview_config("lstm", optimize="both", sequence_length=120)

print("\nHeterogeneous ensemble:")
preview_config(["xgboost", "lstm", "patchtst"], optimize="features")

# -----------------------------------------------------------------------------
# 6. ACTUAL TRAINING EXAMPLES (commented out - run individually)
# -----------------------------------------------------------------------------
print("\n\n6. TRAINING EXAMPLES (code snippets)")
print("-" * 40)

examples = """
# Example 1: Absolute beginner - just works!
results = train("xgboost")

# Example 2: Pick symbol and horizon
results = train("xgboost", symbol="MGC", horizon=20)

# Example 3: Multiple horizons
results = train("xgboost", horizon=[5, 10, 15, 20])

# Example 4: Want optimization
results = train("xgboost", optimize="hyperparameters")

# Example 5: Multiple models
results = train(["xgboost", "lstm"])

# Example 6: Multiple models with ensemble
results = train(["xgboost", "lstm"], ensemble=True)

# Example 7: Full heterogeneous ensemble with optimization
results = train(
    ["xgboost", "lstm", "patchtst"],
    symbol="MES",
    horizon=20,
    optimize="both",
    ensemble=True,
    meta_learner="ridge_meta",
)

# Example 8: Override a smart default
results = train("lstm", sequence_length=120)

# Example 9: Per-model overrides
results = train(
    ["xgboost", "lstm"],
    xgboost={"timeframe": "5min"},  # Override xgboost's timeframe
    lstm={"sequence_length": 120},   # Override lstm's sequence length
)
"""

print(examples)

# -----------------------------------------------------------------------------
# 7. UNDERSTANDING THE SYSTEM
# -----------------------------------------------------------------------------
print("\n\n7. HOW IT WORKS")
print("-" * 40)

explanation = """
The system is SMART. You make simple choices, it figures out the details.

USER CHOICES (what you decide):
  1. Which model(s)?     - "xgboost", "lstm", ["xgboost", "lstm"]
  2. Which symbol?       - "MES", "MGC"
  3. Which horizon(s)?   - 5, 10, 15, 20
  4. Optimize what?      - "none", "features", "hyperparameters", "both"
  5. Build ensemble?     - True/False

SMART DEFAULTS (what the system figures out for you):

  Model Type        Timeframe   Features        Seq Len   Batch
  ---------------   ---------   -------------   -------   -----
  Boosting          15min       ~100 indicators   N/A      N/A
  Neural (RNN)       5min       ~80 + wavelets    60       256
  Transformers       1min       Raw OHLCV only    96       64

WHY THESE DEFAULTS?
  - Boosting: Works best with aggregated data (15min), loves engineered features
  - RNNs: Need finer granularity (5min), benefit from wavelets
  - Transformers: Learn from raw data (1min), no pre-engineering needed

OVERRIDE ANYTHING:
  train("lstm", sequence_length=120)  # Override the default 60
"""

print(explanation)

print("\n" + "=" * 60)
print("Ready to train? Just call train()!")
print("=" * 60)
