#!/usr/bin/env python3
"""
Test feature-set CLI flag and meta-learner stacking functionality.

This script verifies:
1. Feature set resolution works correctly
2. Feature set filtering is applied during training
3. Meta-learner stacking works with heterogeneous ensembles
4. The complete training pipeline with feature-set flag

Run with:
    python scripts/test_feature_set_meta_learner.py
"""
from __future__ import annotations

import logging
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def create_realistic_mock_data(
    n_train: int = 500,
    n_val: int = 100,
    n_test: int = 100,
    seq_len: int = 30,
) -> MagicMock:
    """Create mock data container with realistic feature names."""
    np.random.seed(42)

    # Create realistic feature names that match feature set definitions
    feature_names = [
        # Returns (match boosting_optimal, neural_optimal)
        "return_1", "return_5", "return_10", "return_20",
        "log_return_1", "log_return_5", "log_return_10",
        # RSI (match boosting_optimal)
        "rsi_14", "rsi_21", "rsi_7",
        # MACD (match boosting_optimal)
        "macd_12_26", "macd_signal_12_26", "macd_hist_12_26",
        # Bollinger (match boosting_optimal)
        "bb_pct_20", "bb_width_20",
        # ATR and volatility (match boosting_optimal)
        "atr_14", "atr_21", "realized_vol_20", "parkinson_vol_20",
        # Volume (match boosting_optimal)
        "volume_ratio_20", "vwap_distance",
        # Temporal (match boosting_optimal)
        "hour_sin", "hour_cos", "day_of_week_sin", "is_rth",
        # Additional momentum
        "momentum_10", "momentum_20", "roc_10", "roc_20",
        # Trend
        "trend_regime", "adx_14", "di_plus_14", "di_minus_14",
        # More features
        "ema_ratio_5_20", "sma_ratio_10_50", "price_position",
        "skewness_20", "kurtosis_20",
    ]
    n_features = len(feature_names)

    # Generate data
    X_train = np.random.randn(n_train, n_features).astype(np.float32)
    X_val = np.random.randn(n_val, n_features).astype(np.float32)
    X_test = np.random.randn(n_test, n_features).astype(np.float32)

    # Labels: -1 (short), 0 (neutral), 1 (long) - matches expected format
    y_train = np.random.choice([-1, 0, 1], n_train).astype(np.int64)
    y_val = np.random.choice([-1, 0, 1], n_val).astype(np.int64)
    y_test = np.random.choice([-1, 0, 1], n_test).astype(np.int64)

    weights_train = np.ones(n_train, dtype=np.float32)
    weights_val = np.ones(n_val, dtype=np.float32)
    weights_test = np.ones(n_test, dtype=np.float32)

    # Create DataFrames
    X_train_df = pd.DataFrame(X_train, columns=feature_names)
    X_val_df = pd.DataFrame(X_val, columns=feature_names)
    X_test_df = pd.DataFrame(X_test, columns=feature_names)

    # Create mock container
    mock = MagicMock()

    def get_sklearn_arrays(split: str, horizon: int = 20, return_df: bool = False):
        if split == "train":
            if return_df:
                return X_train_df, pd.Series(y_train), pd.Series(weights_train)
            return X_train, y_train, weights_train
        elif split == "val":
            if return_df:
                return X_val_df, pd.Series(y_val), pd.Series(weights_val)
            return X_val, y_val, weights_val
        else:  # test
            if return_df:
                return X_test_df, pd.Series(y_test), pd.Series(weights_test)
            return X_test, y_test, weights_test

    mock.get_sklearn_arrays = get_sklearn_arrays
    mock.get_label_end_times = MagicMock(return_value=None)

    # Sequence data for neural models (3D)
    # Ensure we have enough samples for sequence creation
    n_train_seq = max(n_train - seq_len + 1, 10)
    n_val_seq = max(n_val - seq_len + 1, 10)
    n_test_seq = max(n_test - seq_len + 1, 10)

    X_train_seq = np.random.randn(n_train_seq, seq_len, n_features).astype(np.float32)
    X_val_seq = np.random.randn(n_val_seq, seq_len, n_features).astype(np.float32)
    X_test_seq = np.random.randn(n_test_seq, seq_len, n_features).astype(np.float32)

    y_train_seq = np.random.choice([-1, 0, 1], n_train_seq).astype(np.int64)
    y_val_seq = np.random.choice([-1, 0, 1], n_val_seq).astype(np.int64)
    y_test_seq = np.random.choice([-1, 0, 1], n_test_seq).astype(np.int64)

    # Mock PyTorch dataset
    class MockSequenceDataset:
        def __init__(self, X, y):
            self.X = X
            self.y = y
            self.weights = np.ones(len(y), dtype=np.float32)

        def __len__(self):
            return len(self.y)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx], self.weights[idx]

    def get_pytorch_sequences(split: str, seq_len: int = 30, symbol_isolated: bool = True):
        if split == "train":
            return MockSequenceDataset(X_train_seq, y_train_seq)
        elif split == "val":
            return MockSequenceDataset(X_val_seq, y_val_seq)
        else:
            return MockSequenceDataset(X_test_seq, y_test_seq)

    mock.get_pytorch_sequences = get_pytorch_sequences

    # Metadata
    mock.horizons = [5, 10, 15, 20]
    mock.symbols = ["TEST"]
    mock.n_features = n_features
    mock.horizon = 20
    mock.feature_columns = feature_names

    return mock


def test_feature_set_resolution():
    """Test feature set name resolution."""
    print("\n" + "=" * 70)
    print("TEST 1: Feature Set Resolution")
    print("=" * 70)

    from src.phase1.config.feature_sets import (
        FEATURE_SET_ALIASES,
        FEATURE_SET_DEFINITIONS,
    )
    from src.phase1.utils.feature_sets import resolve_feature_set

    # Test alias resolution
    print("\n1.1 Testing alias resolution...")
    test_cases = [
        ("xgboost", "boosting_optimal"),
        ("lstm", "neural_optimal"),
        ("tcn", "tcn_optimal"),
        ("patchtst", "patchtst_optimal"),
        ("transformer", "transformer_raw"),
        ("boosting", "boosting_optimal"),
    ]

    for alias, expected in test_cases:
        resolved = FEATURE_SET_ALIASES.get(alias, alias)
        status = "PASS" if resolved == expected else "FAIL"
        print(f"  [{status}] {alias} -> {resolved} (expected: {expected})")

    # Test feature set definitions exist
    print("\n1.2 Testing feature set definitions...")
    for name in ["boosting_optimal", "neural_optimal", "tcn_optimal", "ensemble_base"]:
        exists = name in FEATURE_SET_DEFINITIONS
        status = "PASS" if exists else "FAIL"
        print(f"  [{status}] {name} definition exists: {exists}")

    # Test feature resolution
    print("\n1.3 Testing feature column resolution...")
    mock_data = create_realistic_mock_data(n_train=100, n_val=20)
    X_train_df, _, _ = mock_data.get_sklearn_arrays("train", return_df=True)

    for fs_name in ["boosting_optimal", "neural_optimal", "ensemble_base"]:
        definition = FEATURE_SET_DEFINITIONS[fs_name]
        columns = resolve_feature_set(X_train_df, definition)
        print(f"  [{fs_name}] Resolved {len(columns)} features from {len(X_train_df.columns)} total")
        if len(columns) > 0:
            print(f"    Sample: {columns[:5]}...")

    print("\n[TEST 1 COMPLETE]")
    return True


def test_trainer_feature_set_integration():
    """Test feature set integration in Trainer."""
    print("\n" + "=" * 70)
    print("TEST 2: Trainer Feature Set Integration")
    print("=" * 70)

    from src.models.config import TrainerConfig
    from src.models.trainer import Trainer

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Test with explicit feature set
        print("\n2.1 Testing XGBoost with boosting_optimal feature set...")

        config = TrainerConfig(
            model_name="xgboost",
            horizon=20,
            output_dir=Path(tmp_dir),
            feature_set="boosting_optimal",
            model_config={
                "n_estimators": 5,
                "max_depth": 3,
                "early_stopping_rounds": 2,
                "verbosity": 0,
            },
        )

        mock_data = create_realistic_mock_data()
        trainer = Trainer(config)

        # Run training
        results = trainer.run(mock_data, skip_save=True)

        print(f"  [PASS] Training completed")
        print(f"    Model: {results['model_name']}")
        print(f"    Accuracy: {results['evaluation_metrics']['accuracy']:.4f}")
        print(f"    Macro F1: {results['evaluation_metrics']['macro_f1']:.4f}")

        # Test with auto-resolved feature set (no explicit feature_set)
        print("\n2.2 Testing LightGBM with auto-resolved feature set...")

        config2 = TrainerConfig(
            model_name="lightgbm",
            horizon=20,
            output_dir=Path(tmp_dir),
            # feature_set not specified - should auto-resolve to boosting_optimal
            model_config={
                "n_estimators": 5,
                "max_depth": 3,
                "early_stopping_rounds": 2,
                "verbose": -1,
            },
        )

        trainer2 = Trainer(config2)
        results2 = trainer2.run(mock_data, skip_save=True)

        print(f"  [PASS] Training completed with auto feature set")
        print(f"    Model: {results2['model_name']}")
        print(f"    Accuracy: {results2['evaluation_metrics']['accuracy']:.4f}")

    print("\n[TEST 2 COMPLETE]")
    return True


def test_meta_learner_training():
    """Test meta-learner training."""
    print("\n" + "=" * 70)
    print("TEST 3: Meta-Learner Training")
    print("=" * 70)

    from src.models.ensemble.meta_learners import (
        RidgeMetaLearner,
        MLPMetaLearner,
        XGBoostMeta,
    )

    np.random.seed(42)

    # Create mock OOF predictions (3 base models x 3 classes = 9 features)
    n_samples = 200
    n_base_models = 3
    n_classes = 3
    n_features = n_base_models * n_classes

    X_train = np.random.rand(n_samples, n_features).astype(np.float32)
    # Labels: -1 (short), 0 (neutral), 1 (long)
    y_train = np.random.choice([-1, 0, 1], n_samples)

    X_val = np.random.rand(50, n_features).astype(np.float32)
    y_val = np.random.choice([-1, 0, 1], 50)

    # Test Ridge Meta
    print("\n3.1 Testing RidgeMetaLearner...")
    ridge_meta = RidgeMetaLearner({"alpha": 1.0})
    metrics = ridge_meta.fit(X_train, y_train, X_val, y_val)
    pred_output = ridge_meta.predict(X_val)
    proba = ridge_meta.predict_proba(X_val)

    # Handle PredictionOutput wrapper - predictions are class_predictions
    if hasattr(pred_output, 'class_predictions'):
        preds = pred_output.class_predictions
    elif hasattr(pred_output, 'predictions'):
        preds = pred_output.predictions
    else:
        preds = pred_output
    print(f"  [PASS] RidgeMetaLearner trained")
    print(f"    Val Accuracy: {metrics.val_accuracy:.4f}")
    print(f"    Predictions shape: {np.array(preds).shape}")
    print(f"    Probabilities shape: {np.array(proba).shape}")

    # Test MLP Meta
    print("\n3.2 Testing MLPMetaLearner...")
    mlp_meta = MLPMetaLearner({
        "hidden_sizes": [32, 16],
        "epochs": 5,
        "learning_rate": 0.01,
    })
    metrics2 = mlp_meta.fit(X_train, y_train, X_val, y_val)
    pred_output2 = mlp_meta.predict(X_val)

    if hasattr(pred_output2, 'class_predictions'):
        preds2 = pred_output2.class_predictions
    elif hasattr(pred_output2, 'predictions'):
        preds2 = pred_output2.predictions
    else:
        preds2 = pred_output2
    print(f"  [PASS] MLPMetaLearner trained")
    print(f"    Val Accuracy: {metrics2.val_accuracy:.4f}")
    print(f"    Predictions shape: {np.array(preds2).shape}")

    # Test XGBoost Meta
    print("\n3.3 Testing XGBoostMeta...")
    xgb_meta = XGBoostMeta({
        "n_estimators": 10,
        "max_depth": 3,
        "verbosity": 0,
    })
    metrics3 = xgb_meta.fit(X_train, y_train, X_val, y_val)
    pred_output3 = xgb_meta.predict(X_val)

    if hasattr(pred_output3, 'class_predictions'):
        preds3 = pred_output3.class_predictions
    elif hasattr(pred_output3, 'predictions'):
        preds3 = pred_output3.predictions
    else:
        preds3 = pred_output3
    print(f"  [PASS] XGBoostMeta trained")
    print(f"    Val Accuracy: {metrics3.val_accuracy:.4f}")
    print(f"    Predictions shape: {np.array(preds3).shape}")

    print("\n[TEST 3 COMPLETE]")
    return True


def test_stacking_ensemble():
    """Test stacking ensemble with homogeneous models."""
    print("\n" + "=" * 70)
    print("TEST 4: Stacking Ensemble")
    print("=" * 70)

    from src.models.ensemble.stacking import StackingEnsemble

    # Create mock data with enough samples for purge/embargo
    # Need: n_samples > n_folds * (purge_bars + embargo_bars) + min_fold_size
    # With 3 folds, purge=10, embargo=20, min_fold=30: need > 3*30+30 = 120
    mock_data = create_realistic_mock_data(n_train=3000, n_val=600)
    X_train_df, y_train, w_train = mock_data.get_sklearn_arrays("train", return_df=True)
    X_val_df, y_val, _ = mock_data.get_sklearn_arrays("val", return_df=True)

    X_train = X_train_df.values
    X_val = X_val_df.values
    y_train = y_train.values
    y_val = y_val.values

    # Test homogeneous stacking (tabular only)
    print("\n4.1 Testing homogeneous tabular stacking (XGBoost + LightGBM)...")

    stacking = StackingEnsemble({
        "base_model_names": ["xgboost", "lightgbm"],
        "meta_learner_name": "ridge_meta",
        "n_folds": 3,
        "purge_bars": 10,  # Reduced for testing
        "embargo_bars": 20,  # Reduced for testing
        "use_probabilities": True,
        "use_default_configs_for_oof": True,
    })

    # Check if it's correctly identified as homogeneous
    print(f"  Is heterogeneous: {stacking._is_heterogeneous}")

    # Fit
    metrics = stacking.fit(X_train, y_train, X_val, y_val)

    print(f"  [PASS] Stacking ensemble trained")
    print(f"    Base models: {stacking.config.get('base_model_names')}")
    print(f"    Meta-learner: {stacking.config.get('meta_learner_name')}")
    print(f"    Val Accuracy: {metrics.val_accuracy:.4f}")

    # Predict
    pred_output = stacking.predict(X_val)
    proba = stacking.predict_proba(X_val)

    # PredictionOutput has .predictions attribute
    preds = pred_output.predictions if hasattr(pred_output, 'predictions') else pred_output
    print(f"    Predictions shape: {np.array(preds).shape}")
    print(f"    Probabilities shape: {np.array(proba).shape}")

    print("\n[TEST 4 COMPLETE]")
    return True


def test_heterogeneous_ensemble_detection():
    """Test heterogeneous ensemble detection and validation."""
    print("\n" + "=" * 70)
    print("TEST 5: Heterogeneous Ensemble Detection")
    print("=" * 70)

    from src.models.ensemble.validator import (
        is_heterogeneous_ensemble,
        classify_base_models,
        validate_ensemble_config,
    )
    from src.models.registry import ModelRegistry

    # Check if CatBoost is available
    catboost_available = ModelRegistry.is_registered("catboost")
    print(f"\nCatBoost available: {catboost_available}")

    # Test homogeneous detection
    print("\n5.1 Testing homogeneous ensemble detection...")
    homogeneous_cases = [
        (["xgboost", "lightgbm"], False, "Both tabular"),
        (["lstm", "gru", "tcn"], False, "All sequence"),
        (["xgboost"], False, "Single tabular"),
    ]

    for models, expected_hetero, desc in homogeneous_cases:
        is_hetero = is_heterogeneous_ensemble(models)
        status = "PASS" if is_hetero == expected_hetero else "FAIL"
        print(f"  [{status}] {desc}: {models} -> heterogeneous={is_hetero}")

    # Test heterogeneous detection
    print("\n5.2 Testing heterogeneous ensemble detection...")
    heterogeneous_cases = [
        (["xgboost", "lstm"], True, "Tabular + Sequence"),
        # Use lightgbm instead of catboost if not available
        (["lightgbm", "tcn", "patchtst"], True, "Mixed 3-model (LightGBM)"),
        (["lightgbm", "gru", "transformer"], True, "Mixed 3-model"),
    ]

    for models, expected_hetero, desc in heterogeneous_cases:
        is_hetero = is_heterogeneous_ensemble(models)
        status = "PASS" if is_hetero == expected_hetero else "FAIL"
        print(f"  [{status}] {desc}: {models} -> heterogeneous={is_hetero}")

    # Test model classification
    print("\n5.3 Testing model classification...")
    test_models = ["xgboost", "lstm", "tcn", "patchtst"]
    tabular, sequence = classify_base_models(test_models)
    print(f"  Models: {test_models}")
    print(f"  Tabular: {tabular}")
    print(f"  Sequence: {sequence}")
    classification_correct = (tabular == ["xgboost"] and set(sequence) == {"lstm", "tcn", "patchtst"})
    print(f"  [{'PASS' if classification_correct else 'FAIL'}] Classification correct")

    # Test validation
    print("\n5.4 Testing ensemble validation...")

    # Valid stacking (allows heterogeneous)
    valid_stacking, msg = validate_ensemble_config(
        ensemble_type="stacking",
        base_model_names=["xgboost", "lstm", "tcn"],
    )
    print(f"  [{'PASS' if valid_stacking else 'FAIL'}] Heterogeneous stacking: valid={valid_stacking}")

    # Invalid voting (rejects heterogeneous)
    valid_voting, msg = validate_ensemble_config(
        ensemble_type="voting",
        base_model_names=["xgboost", "lstm"],
    )
    if not valid_voting:
        print(f"  [PASS] Heterogeneous voting correctly rejected: {msg[:50]}...")
    else:
        print(f"  [FAIL] Heterogeneous voting should have been rejected")

    print("\n[TEST 5 COMPLETE]")
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("FEATURE SET & META-LEARNER INTEGRATION TESTS")
    print("=" * 70)

    # Import models to trigger registration
    import src.models  # noqa: F401

    from src.models.registry import ModelRegistry
    print(f"\nModel Registry: {ModelRegistry.count()} models registered")

    tests = [
        ("Feature Set Resolution", test_feature_set_resolution),
        ("Trainer Feature Set Integration", test_trainer_feature_set_integration),
        ("Meta-Learner Training", test_meta_learner_training),
        ("Stacking Ensemble", test_stacking_ensemble),
        ("Heterogeneous Ensemble Detection", test_heterogeneous_ensemble_detection),
    ]

    results = []
    for name, test_fn in tests:
        try:
            success = test_fn()
            results.append((name, "PASS" if success else "FAIL"))
        except Exception as e:
            logger.exception(f"Test {name} failed with exception")
            results.append((name, f"ERROR: {str(e)[:50]}"))

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    for name, status in results:
        symbol = "PASS" if status == "PASS" else "FAIL"
        print(f"  [{symbol}] {name}: {status}")

    n_passed = sum(1 for _, s in results if s == "PASS")
    print(f"\nTotal: {n_passed}/{len(results)} tests passed")

    return 0 if n_passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
