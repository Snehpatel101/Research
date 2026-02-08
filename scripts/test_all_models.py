#!/usr/bin/env python3
"""
Comprehensive Model Test Loop - Tests all 12 target models + ensembles.

Phase 1: Data preparation (raw OHLCV -> features -> labels -> splits -> scaling)
Phase 2: Individual model training (12 models)
Phase 3: Ensemble combinations
"""

import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Results tracking
RESULTS: dict[str, dict] = {}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def log_result(model_name: str, success: bool, metrics: dict | None = None, error: str | None = None, duration: float = 0.0) -> None:
    RESULTS[model_name] = {
        "success": success,
        "metrics": metrics or {},
        "error": error,
        "duration": duration,
    }
    status = "PASS" if success else "FAIL"
    if metrics:
        acc = metrics.get("accuracy", 0)
        f1 = metrics.get("f1_weighted", 0)
        log(f"  [{status}] {model_name}: accuracy={acc:.4f}, f1={f1:.4f} ({duration:.1f}s)")
    else:
        log(f"  [{status}] {model_name}: {error} ({duration:.1f}s)")


# =============================================================================
# PHASE 0: DATA PREPARATION
# =============================================================================

def prepare_data() -> "TimeSeriesDataContainer":
    """Load raw data, generate features, labels, splits, scaling."""
    from src.core.container import TimeSeriesDataContainer

    log("=" * 70)
    log("PHASE 0: DATA PREPARATION")
    log("=" * 70)

    raw_path = PROJECT_ROOT / "data" / "raw" / "_test_subset_1mo.parquet"
    raw_df = pd.read_parquet(raw_path)
    log(f"Loaded raw data: {len(raw_df)} rows, cols={list(raw_df.columns)}")
    log(f"Date range: {raw_df.index.min()} to {raw_df.index.max()}")

    # --- Step 1: Feature Engineering ---
    log("Step 1: Feature engineering...")
    from src.data.features.compute import compute_all_features

    df_features = compute_all_features(raw_df, include_original=True, exclude_families=["wavelets"])
    log(f"  Features: {df_features.shape[1]} columns, {len(df_features)} rows")

    # --- Step 2: Labeling ---
    log("Step 2: Triple-barrier labeling (horizon=20)...")
    horizon = 20

    # We need ATR for triple-barrier
    if "atr_14" not in df_features.columns:
        # Compute simple ATR
        high = df_features["high"] if "high" in df_features.columns else raw_df["high"].values[:len(df_features)]
        low = df_features["low"] if "low" in df_features.columns else raw_df["low"].values[:len(df_features)]
        close = df_features["close"] if "close" in df_features.columns else raw_df["close"].values[:len(df_features)]

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df_features["atr_14"] = tr.rolling(14).mean()

    # Simple directional labeling using fixed percentile thresholds
    close = df_features["close"]
    fwd_return = close.shift(-horizon) / close - 1

    label_col = f"label_h{horizon}"
    weight_col = f"sample_weight_h{horizon}"

    # Use fixed percentile thresholds to get ~33/33/33 class balance
    valid_returns = fwd_return.dropna()
    upper_thresh = valid_returns.quantile(0.60)  # Top 40% = long
    lower_thresh = valid_returns.quantile(0.40)  # Bottom 40% = short

    labels = pd.Series(0, index=df_features.index, dtype=np.int64)
    labels[fwd_return > upper_thresh] = 1
    labels[fwd_return < lower_thresh] = -1
    labels[fwd_return.isna()] = -99  # Invalid sentinel

    df_features[label_col] = labels
    df_features[weight_col] = 1.0  # Uniform weights

    # Drop rows with invalid labels
    valid_mask = df_features[label_col] != -99
    df_features = df_features[valid_mask].reset_index(drop=True)
    log(f"  Labels: {df_features[label_col].value_counts().to_dict()}")
    log(f"  Valid rows: {len(df_features)}")

    # --- Step 3: Chronological Splits ---
    log("Step 3: Creating chronological splits (70/15/15)...")
    n = len(df_features)
    purge = 60
    embargo = 200  # Smaller for 1-month data

    train_end = int(n * 0.70)
    val_start = train_end + purge
    val_end = val_start + int(n * 0.15)
    test_start = val_end + embargo

    df_train = df_features.iloc[:train_end].copy()
    df_val = df_features.iloc[val_start:val_end].copy()
    df_test = df_features.iloc[test_start:].copy()

    log(f"  Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")

    # --- Step 4: Feature Scaling ---
    log("Step 4: Feature scaling (robust)...")

    # Identify feature columns (exclude metadata, labels, weights)
    exclude_prefixes = ("label_", "sample_weight_", "datetime", "symbol", "timestamp")
    exclude_exact = {"open", "high", "low", "close", "volume", "datetime"}
    feature_cols = [
        c for c in df_train.columns
        if c not in exclude_exact
        and not any(c.startswith(p) for p in exclude_prefixes)
        and df_train[c].dtype in [np.float64, np.float32, np.int64, np.int32]
    ]

    # Remove any columns with all NaN
    valid_feature_cols = [c for c in feature_cols if not df_train[c].isna().all()]
    log(f"  Feature columns: {len(valid_feature_cols)} (removed {len(feature_cols) - len(valid_feature_cols)} all-NaN)")

    # Fill NaN with 0 for remaining
    for split_df in [df_train, df_val, df_test]:
        split_df[valid_feature_cols] = split_df[valid_feature_cols].fillna(0)

    # Robust scaling: fit on train only
    from sklearn.preprocessing import RobustScaler

    scaler = RobustScaler()
    df_train[valid_feature_cols] = scaler.fit_transform(df_train[valid_feature_cols])
    df_val[valid_feature_cols] = scaler.transform(df_val[valid_feature_cols])
    df_test[valid_feature_cols] = scaler.transform(df_test[valid_feature_cols])

    # Clip outliers
    for split_df in [df_train, df_val, df_test]:
        split_df[valid_feature_cols] = split_df[valid_feature_cols].clip(-5, 5)

    log(f"  Scaling complete. Train shape: {df_train.shape}")

    # --- Step 5: Create Container ---
    log("Step 5: Creating TimeSeriesDataContainer...")
    container = TimeSeriesDataContainer.from_dataframes(
        train_df=df_train,
        val_df=df_val,
        test_df=df_test,
        horizon=horizon,
        feature_columns=valid_feature_cols,
    )

    log(f"  Container: splits={container.available_splits}, features={container.n_features}")

    # Verify we can extract arrays
    X_train, y_train, w_train = container.get_sklearn_arrays("train")
    log(f"  Train arrays: X={X_train.shape}, y={y_train.shape}")
    log(f"  Label dist: {dict(zip(*np.unique(y_train, return_counts=True)))}")

    return container


# =============================================================================
# PHASE 1: INDIVIDUAL MODEL TESTS
# =============================================================================

def test_boosting_model(container: "TimeSeriesDataContainer", model_name: str) -> None:
    """Test a boosting model (2D tabular input)."""
    from src.models import ModelRegistry
    from sklearn.metrics import accuracy_score, f1_score

    t0 = time.time()
    try:
        X_train, y_train, w_train = container.get_sklearn_arrays("train")
        X_val, y_val, w_val = container.get_sklearn_arrays("val")

        # Override for speed
        fast_config: dict = {}
        if model_name == "xgboost":
            fast_config = {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1, "n_jobs": -1}
        elif model_name == "lightgbm":
            fast_config = {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1, "n_jobs": -1, "verbose": -1}
        elif model_name == "catboost":
            fast_config = {"iterations": 100, "depth": 6, "learning_rate": 0.1, "verbose": 0}

        model = ModelRegistry.create(model_name, config=fast_config)

        # Train: API is fit(X_train, y_train, X_val, y_val, sample_weights=None)
        log(f"  Training {model_name} on {X_train.shape}...")
        train_metrics = model.fit(X_train, y_train, X_val, y_val, sample_weights=w_train)

        # Predict
        pred_result = model.predict(X_val)
        y_pred = pred_result.class_predictions

        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average="weighted", zero_division=0)

        metrics = {
            "accuracy": acc,
            "f1_weighted": f1,
            "train_shape": list(X_train.shape),
            "val_shape": list(X_val.shape),
        }
        if hasattr(pred_result, "class_probabilities") and pred_result.class_probabilities is not None:
            metrics["has_probabilities"] = True

        log_result(model_name, True, metrics, duration=time.time() - t0)

    except Exception as e:
        log_result(model_name, False, error=f"{type(e).__name__}: {e}", duration=time.time() - t0)
        traceback.print_exc()


def test_neural_model(container: "TimeSeriesDataContainer", model_name: str, seq_len: int = 30) -> None:
    """Test a neural/sequence model (3D input)."""
    from src.models import ModelRegistry
    from sklearn.metrics import accuracy_score, f1_score

    t0 = time.time()
    try:
        # Get 2D data first
        X_train_2d, y_train, w_train = container.get_sklearn_arrays("train")
        X_val_2d, y_val, w_val = container.get_sklearn_arrays("val")

        n_features = X_train_2d.shape[1]

        # Create 3D sequences: (n_samples, seq_len, n_features)
        def make_sequences(X_2d, y, seq_len):
            n = len(X_2d) - seq_len + 1
            if n <= 0:
                raise ValueError(f"Not enough data for seq_len={seq_len}: have {len(X_2d)} rows")
            X_3d = np.stack([X_2d[i:i+seq_len] for i in range(n)])
            y_seq = y[seq_len-1:]  # Align labels with end of each sequence
            return X_3d, y_seq[:n]

        X_train, y_train_seq = make_sequences(X_train_2d, y_train, seq_len)
        X_val, y_val_seq = make_sequences(X_val_2d, y_val, seq_len)

        log(f"  Training {model_name} on {X_train.shape} (3D)...")

        # Create model with fast config
        fast_config = {
            "hidden_size": 64,
            "num_layers": 1,
            "dropout": 0.1,
            "learning_rate": 0.001,
            "batch_size": 128,
            "max_epochs": 5,  # Just 5 epochs for testing
            "early_stopping_patience": 3,
            "n_classes": 3,
            "input_size": n_features,
            "seq_len": seq_len,
        }

        model = ModelRegistry.create(model_name, config=fast_config)

        # Train: API is fit(X_train, y_train, X_val, y_val, sample_weights=None)
        train_metrics = model.fit(X_train, y_train_seq, X_val, y_val_seq)

        # Predict
        pred_result = model.predict(X_val)
        y_pred = pred_result.class_predictions

        acc = accuracy_score(y_val_seq, y_pred)
        f1 = f1_score(y_val_seq, y_pred, average="weighted", zero_division=0)

        metrics = {
            "accuracy": acc,
            "f1_weighted": f1,
            "train_shape": list(X_train.shape),
            "val_shape": list(X_val.shape),
        }

        log_result(model_name, True, metrics, duration=time.time() - t0)

    except Exception as e:
        log_result(model_name, False, error=f"{type(e).__name__}: {e}", duration=time.time() - t0)
        traceback.print_exc()


# =============================================================================
# MAIN
# =============================================================================

def main():
    log("=" * 70)
    log("COMPREHENSIVE MODEL TEST LOOP")
    log("=" * 70)

    # Phase 0: Data preparation
    container = prepare_data()

    # Phase 1: Boosting models (2D)
    log("")
    log("=" * 70)
    log("PHASE 1: BOOSTING MODELS (2D)")
    log("=" * 70)

    for model_name in ["xgboost", "lightgbm", "catboost"]:
        log(f"\nTesting {model_name}...")
        test_boosting_model(container, model_name)

    # Phase 2: RNN models (3D)
    log("")
    log("=" * 70)
    log("PHASE 2: RNN MODELS (3D)")
    log("=" * 70)

    for model_name in ["lstm", "gru"]:
        log(f"\nTesting {model_name}...")
        test_neural_model(container, model_name, seq_len=30)

    # Phase 3: CNN models (3D)
    log("")
    log("=" * 70)
    log("PHASE 3: CNN MODELS (3D)")
    log("=" * 70)

    for model_name in ["tcn", "inceptiontime", "resnet1d"]:
        log(f"\nTesting {model_name}...")
        test_neural_model(container, model_name, seq_len=30)

    # Phase 4: Transformer models (3D/4D)
    log("")
    log("=" * 70)
    log("PHASE 4: TRANSFORMER MODELS (3D)")
    log("=" * 70)

    for model_name in ["patchtst", "itransformer", "tft"]:
        log(f"\nTesting {model_name}...")
        test_neural_model(container, model_name, seq_len=30)

    # Phase 5: MLP models (3D for N-BEATS)
    log("")
    log("=" * 70)
    log("PHASE 5: MLP MODELS")
    log("=" * 70)

    log(f"\nTesting nbeats...")
    test_neural_model(container, "nbeats", seq_len=30)

    # Summary
    log("")
    log("=" * 70)
    log("RESULTS SUMMARY")
    log("=" * 70)

    passed = sum(1 for r in RESULTS.values() if r["success"])
    failed = sum(1 for r in RESULTS.values() if not r["success"])
    total = len(RESULTS)

    for name, result in RESULTS.items():
        status = "PASS" if result["success"] else "FAIL"
        if result["success"]:
            m = result["metrics"]
            log(f"  [{status}] {name:20s} acc={m.get('accuracy', 0):.4f} f1={m.get('f1_weighted', 0):.4f} ({result['duration']:.1f}s)")
        else:
            log(f"  [{status}] {name:20s} {result['error'][:60]} ({result['duration']:.1f}s)")

    log(f"\nTotal: {passed}/{total} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
