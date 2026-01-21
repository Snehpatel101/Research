#!/usr/bin/env python3
"""
Model Performance Diagnostic Script.

Diagnoses coin-flip performance issues in ML models by running comprehensive tests:
1. Overfit test - Can the model memorize a tiny dataset?
2. Baseline comparisons - Compare against naive strategies
3. Label distribution analysis - Check class balance and autocorrelation
4. Confusion matrix analysis - Precision/recall breakdown
5. Time split validation - Verify chronological ordering

Usage:
    # Quick diagnostic with XGBoost
    python scripts/diagnose_model_performance.py --model xgboost --horizon 20

    # Full diagnostic with custom data path
    python scripts/diagnose_model_performance.py --model lstm --horizon 20 \
        --data-dir data/splits/scaled --seq-len 30

    # Run only specific tests
    python scripts/diagnose_model_performance.py --model xgboost --horizon 20 \
        --tests overfit,baselines

Requirements:
    - Phase 1 data in data/splits/scaled/
    - Model registered in ModelRegistry
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# LOGGING SETUP
# =============================================================================


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for diagnostics."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Reduce noise from libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "splits" / "scaled"
LABEL_MAP = {-1: "short", 0: "neutral", 1: "long"}
OVERFIT_SAMPLES = 1000  # Small sample for overfit test
OVERFIT_LOSS_THRESHOLD = 0.5  # Loss should be well below 0.69 (coin flip)
COIN_FLIP_ACCURACY = 0.333  # 3-class random baseline


# =============================================================================
# DIAGNOSTIC FUNCTIONS
# =============================================================================


def create_sequences_from_arrays(
    X: np.ndarray,
    y: np.ndarray,
    seq_len: int,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create sliding window sequences from 2D arrays.

    Args:
        X: Features array (n_samples, n_features)
        y: Labels array (n_samples,)
        seq_len: Sequence length
        stride: Step between sequences

    Returns:
        Tuple of (X_seq, y_seq) where X_seq is (n_sequences, seq_len, n_features)
    """
    n_samples, n_features = X.shape
    n_sequences = (n_samples - seq_len) // stride + 1

    if n_sequences <= 0:
        raise ValueError(f"Not enough samples ({n_samples}) for seq_len={seq_len}")

    # Pre-allocate
    X_seq = np.zeros((n_sequences, seq_len, n_features), dtype=X.dtype)
    y_seq = np.zeros(n_sequences, dtype=y.dtype)

    for i in range(n_sequences):
        start_idx = i * stride
        end_idx = start_idx + seq_len
        X_seq[i] = X[start_idx:end_idx]
        y_seq[i] = y[end_idx - 1]  # Label from last timestep

    return X_seq, y_seq


def run_overfit_test(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_samples: int = OVERFIT_SAMPLES,
    is_sequence: bool = False,
    seq_len: int = 30,
) -> dict[str, Any]:
    """
    Test if model can overfit a tiny dataset.

    If a model cannot drive loss well below 0.69 (binary cross-entropy for 50/50),
    or below 1.1 (log-loss for 3-class uniform), there's a pipeline/model bug.

    Args:
        model_name: Name of the model to test
        X_train: Full training features
        y_train: Full training labels
        n_samples: Number of samples to use (default: 1000)
        is_sequence: Whether model requires sequences
        seq_len: Sequence length for sequence models

    Returns:
        Dict with overfit test results
    """
    from src.models.registry import ModelRegistry

    logger.info(f"Running overfit test with {n_samples} samples...")

    # Use first n_samples (maintaining temporal order)
    X_subset = X_train[:n_samples]
    y_subset = y_train[:n_samples]

    # For sequence models, adjust based on seq_len
    if is_sequence and len(X_subset.shape) == 3:
        # X is already 3D (n_samples, seq_len, n_features)
        pass
    elif is_sequence and len(X_subset.shape) == 2:
        # Need to create sequences from 2D data
        X_subset, y_subset = create_sequences_from_arrays(
            X_subset,
            y_subset,
            seq_len=seq_len,
        )
        # Limit to n_samples sequences
        X_subset = X_subset[:n_samples]
        y_subset = y_subset[:n_samples]

    # Split into train/val (80/20) for the overfit test
    split_idx = int(len(X_subset) * 0.8)
    X_overfit_train = X_subset[:split_idx]
    y_overfit_train = y_subset[:split_idx]
    X_overfit_val = X_subset[split_idx:]
    y_overfit_val = y_subset[split_idx:]

    # Create model with aggressive settings for overfitting
    overfit_config = {
        "max_epochs": 200,
        "early_stopping_patience": 50,
        "n_estimators": 500,
        "max_depth": 12,  # Deep trees to overfit
        "learning_rate": 0.1,  # Higher LR for faster fitting
        "min_child_weight": 1,  # Allow small leaves
        "reg_alpha": 0.0,  # No regularization
        "reg_lambda": 0.0,  # No regularization
        "subsample": 1.0,  # Use all data
        "colsample_bytree": 1.0,  # Use all features
        "dropout": 0.0,  # No dropout
    }

    model = ModelRegistry.create(model_name, config=overfit_config)

    # Train and measure
    start_time = time.time()
    try:
        metrics = model.fit(
            X_train=X_overfit_train,
            y_train=y_overfit_train,
            X_val=X_overfit_val,
            y_val=y_overfit_val,
            sample_weights=None,
            config=overfit_config,
        )
        training_time = time.time() - start_time

        # Get predictions on training data (should be near-perfect if overfitting)
        train_preds = model.predict(X_overfit_train)
        train_accuracy = accuracy_score(y_overfit_train, train_preds.class_predictions)
        train_f1 = f1_score(
            y_overfit_train, train_preds.class_predictions, average="macro", zero_division="warn"
        )

        # Compute cross-entropy loss proxy
        # For 3-class: uniform random = log(3) = 1.1
        # Perfect = 0, coin-flip ~= 1.1
        train_loss = metrics.train_loss

        result = {
            "passed": train_accuracy > 0.9 and train_loss < OVERFIT_LOSS_THRESHOLD,
            "train_accuracy": float(train_accuracy),
            "train_f1": float(train_f1),
            "train_loss": float(train_loss),
            "val_loss": float(metrics.val_loss),
            "epochs_trained": metrics.epochs_trained,
            "training_time_seconds": training_time,
            "n_samples": n_samples,
            "threshold": {
                "min_accuracy": 0.9,
                "max_loss": OVERFIT_LOSS_THRESHOLD,
            },
        }

        if result["passed"]:
            logger.info(
                f"PASSED: Model can overfit (train_acc={train_accuracy:.3f}, "
                f"train_loss={train_loss:.4f})"
            )
        else:
            logger.warning(
                f"FAILED: Model cannot overfit small data (train_acc={train_accuracy:.3f}, "
                f"train_loss={train_loss:.4f}). Possible pipeline/model bug."
            )

    except Exception as e:
        logger.error(f"Overfit test failed with error: {e}")
        result = {
            "passed": False,
            "error": str(e),
            "n_samples": n_samples,
        }

    return result


def compute_baseline_metrics(
    y_true: np.ndarray,
) -> dict[str, dict[str, float]]:
    """
    Compute performance metrics for various baseline strategies.

    Baselines:
    1. Majority class - Always predict most common class
    2. Random uniform - Random 50/50/0 or 33/33/33 predictions
    3. Predict previous - Naive persistence (predict previous label)
    4. Always long - Always predict 1
    5. Always short - Always predict -1
    6. Always neutral - Always predict 0

    Args:
        y_true: True labels (-1, 0, 1)

    Returns:
        Dict mapping baseline name to metrics dict
    """
    logger.info("Computing baseline metrics...")
    n_samples = len(y_true)
    baselines = {}

    # 1. Majority class predictor
    unique, counts = np.unique(y_true, return_counts=True)
    majority_class = unique[np.argmax(counts)]
    y_majority = np.full(n_samples, majority_class)
    baselines["majority_class"] = {
        "accuracy": float(accuracy_score(y_true, y_majority)),
        "f1_macro": float(f1_score(y_true, y_majority, average="macro", zero_division="warn")),
        "description": f"Always predict {LABEL_MAP.get(majority_class, majority_class)}",
    }

    # 2. Random predictor (uniform)
    np.random.seed(42)
    y_random = np.random.choice([-1, 0, 1], size=n_samples)
    baselines["random_uniform"] = {
        "accuracy": float(accuracy_score(y_true, y_random)),
        "f1_macro": float(f1_score(y_true, y_random, average="macro", zero_division="warn")),
        "description": "Random uniform across 3 classes",
    }

    # 3. Predict previous label (naive persistence)
    y_persistence = np.concatenate([[y_true[0]], y_true[:-1]])
    baselines["persistence"] = {
        "accuracy": float(accuracy_score(y_true, y_persistence)),
        "f1_macro": float(f1_score(y_true, y_persistence, average="macro", zero_division="warn")),
        "description": "Predict previous label",
    }

    # 4. Always long
    y_long = np.ones(n_samples, dtype=int)
    baselines["always_long"] = {
        "accuracy": float(accuracy_score(y_true, y_long)),
        "f1_macro": float(f1_score(y_true, y_long, average="macro", zero_division="warn")),
        "description": "Always predict long (1)",
    }

    # 5. Always short
    y_short = np.full(n_samples, -1)
    baselines["always_short"] = {
        "accuracy": float(accuracy_score(y_true, y_short)),
        "f1_macro": float(f1_score(y_true, y_short, average="macro", zero_division="warn")),
        "description": "Always predict short (-1)",
    }

    # 6. Always neutral
    y_neutral = np.zeros(n_samples, dtype=int)
    baselines["always_neutral"] = {
        "accuracy": float(accuracy_score(y_true, y_neutral)),
        "f1_macro": float(f1_score(y_true, y_neutral, average="macro", zero_division="warn")),
        "description": "Always predict neutral (0)",
    }

    return baselines


def analyze_label_distribution(
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray | None = None,
) -> dict[str, Any]:
    """
    Analyze label distribution across splits.

    Checks:
    1. Class balance statistics
    2. Label autocorrelation (persistence)
    3. Distribution per split

    Args:
        y_train: Training labels
        y_val: Validation labels
        y_test: Test labels (optional)

    Returns:
        Dict with distribution analysis results
    """
    logger.info("Analyzing label distribution...")

    result = {
        "splits": {},
        "autocorrelation": {},
        "class_balance": {},
    }

    # Analyze each split
    for name, y in [("train", y_train), ("val", y_val), ("test", y_test)]:
        if y is None:
            continue

        unique, counts = np.unique(y, return_counts=True)
        total = len(y)
        distribution = {LABEL_MAP.get(int(c), str(c)): int(cnt) for c, cnt in zip(unique, counts)}
        percentages = {
            LABEL_MAP.get(int(c), str(c)): float(cnt / total * 100)
            for c, cnt in zip(unique, counts)
        }

        result["splits"][name] = {
            "n_samples": total,
            "distribution": distribution,
            "percentages": percentages,
        }

        # Compute autocorrelation (lag-1)
        if len(y) > 1:
            # Persistence rate: how often does y[t] == y[t-1]?
            persistence = (y[1:] == y[:-1]).mean()
            result["autocorrelation"][name] = {
                "lag1_persistence": float(persistence),
                "description": "Fraction where label[t] == label[t-1]",
            }

    # Overall class balance metric (Gini-like imbalance)
    all_labels = np.concatenate([y_train, y_val] + ([y_test] if y_test is not None else []))
    unique, counts = np.unique(all_labels, return_counts=True)
    props = counts / counts.sum()
    # Perfect balance = 1/3 each, compute deviation
    ideal_prop = 1.0 / len(unique)
    imbalance = np.abs(props - ideal_prop).mean()

    result["class_balance"] = {
        "class_proportions": {
            LABEL_MAP.get(int(c), str(c)): float(p) for c, p in zip(unique, props)
        },
        "imbalance_score": float(imbalance),
        "is_balanced": imbalance < 0.1,  # Less than 10% deviation from uniform
        "description": "Lower imbalance_score = more balanced. 0 = perfectly balanced.",
    }

    return result


def analyze_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> dict[str, Any]:
    """
    Detailed confusion matrix and per-class metrics analysis.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (optional)

    Returns:
        Dict with confusion matrix analysis
    """
    logger.info("Analyzing confusion matrix...")

    # Compute confusion matrix
    labels = [-1, 0, 1]
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Per-class precision, recall, F1
    prec_arr, rec_arr, f1_arr, sup_arr = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division="warn"
    )
    # Cast to arrays (Pyright type stubs are overly restrictive)
    precision = np.asarray(prec_arr)
    recall = np.asarray(rec_arr)
    f1 = np.asarray(f1_arr)
    support = np.asarray(sup_arr)

    # Classification report as dict
    report_dict = classification_report(
        y_true, y_pred, labels=labels, target_names=["short", "neutral", "long"], output_dict=True
    )
    # Cast to dict (Pyright doesn't know output_dict=True returns dict)
    report: dict[str, Any] = dict(report_dict)  # type: ignore[arg-type]

    result = {
        "confusion_matrix": cm.tolist(),
        "labels": ["short", "neutral", "long"],
        "per_class": {
            "short": {
                "precision": float(precision[0]),
                "recall": float(recall[0]),
                "f1": float(f1[0]),
                "support": int(support[0]),
            },
            "neutral": {
                "precision": float(precision[1]),
                "recall": float(recall[1]),
                "f1": float(f1[1]),
                "support": int(support[1]),
            },
            "long": {
                "precision": float(precision[2]),
                "recall": float(recall[2]),
                "f1": float(f1[2]),
                "support": int(support[2]),
            },
        },
        "overall": {
            "accuracy": float(report["accuracy"]),
            "macro_f1": float(report["macro avg"]["f1-score"]),
            "weighted_f1": float(report["weighted avg"]["f1-score"]),
        },
    }

    # Threshold sensitivity analysis (if probabilities available)
    if y_proba is not None:
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
        threshold_analysis = []

        for thresh in thresholds:
            # Convert probabilities to predictions using threshold
            max_proba = np.max(y_proba, axis=1)
            confident_mask = max_proba >= thresh

            if confident_mask.sum() > 0:
                y_pred_thresh = np.argmax(y_proba, axis=1)
                # Map 0,1,2 -> -1,0,1
                y_pred_thresh = y_pred_thresh - 1

                acc = accuracy_score(y_true[confident_mask], y_pred_thresh[confident_mask])
                f1_macro = f1_score(
                    y_true[confident_mask],
                    y_pred_thresh[confident_mask],
                    average="macro",
                    zero_division="warn",
                )
                coverage = confident_mask.mean()
            else:
                acc = 0.0
                f1_macro = 0.0
                coverage = 0.0

            threshold_analysis.append(
                {
                    "threshold": thresh,
                    "accuracy": float(acc),
                    "f1_macro": float(f1_macro),
                    "coverage": float(coverage),
                }
            )

        result["threshold_analysis"] = threshold_analysis

    return result


def validate_time_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame | None = None,
    datetime_col: str = "datetime",
) -> dict[str, Any]:
    """
    Verify train/val/test are chronologically ordered with no leakage.

    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame (optional)
        datetime_col: Name of datetime column

    Returns:
        Dict with time split validation results
    """
    logger.info("Validating time splits...")

    result = {
        "splits": {},
        "ordering_valid": True,
        "leakage_detected": False,
        "issues": [],
    }

    # Get datetime ranges for each split
    splits = {"train": train_df, "val": val_df}
    if test_df is not None:
        splits["test"] = test_df

    for name, df in splits.items():
        if datetime_col in df.columns:
            dt = pd.to_datetime(df[datetime_col])
            result["splits"][name] = {
                "start": str(dt.min()),
                "end": str(dt.max()),
                "n_samples": len(df),
            }
        else:
            result["splits"][name] = {
                "start": "N/A (no datetime column)",
                "end": "N/A",
                "n_samples": len(df),
            }
            result["issues"].append(f"Missing datetime column '{datetime_col}' in {name}")

    # Check chronological ordering
    if datetime_col in train_df.columns and datetime_col in val_df.columns:
        train_end = pd.to_datetime(train_df[datetime_col]).max()
        val_start = pd.to_datetime(val_df[datetime_col]).min()

        if train_end >= val_start:
            result["ordering_valid"] = False
            result["leakage_detected"] = True
            result["issues"].append(
                f"LEAKAGE: Train end ({train_end}) >= Val start ({val_start})"
            )

        if test_df is not None and datetime_col in test_df.columns:
            val_end = pd.to_datetime(val_df[datetime_col]).max()
            test_start = pd.to_datetime(test_df[datetime_col]).min()

            if val_end >= test_start:
                result["ordering_valid"] = False
                result["leakage_detected"] = True
                result["issues"].append(
                    f"LEAKAGE: Val end ({val_end}) >= Test start ({test_start})"
                )

    if result["ordering_valid"]:
        logger.info("Time splits are correctly ordered (no leakage detected)")
    else:
        logger.error("Time split ordering issues detected - possible data leakage!")

    return result


# =============================================================================
# MAIN DIAGNOSTIC RUNNER
# =============================================================================


def run_diagnostics(
    model_name: str,
    horizon: int,
    data_dir: Path,
    seq_len: int = 30,
    tests: list[str] | None = None,
) -> dict[str, Any]:
    """
    Run comprehensive model diagnostics.

    Args:
        model_name: Name of model to diagnose
        horizon: Label horizon
        data_dir: Path to scaled data directory
        seq_len: Sequence length for sequence models
        tests: List of tests to run (default: all)

    Returns:
        Dict with all diagnostic results
    """
    from src.models.registry import ModelRegistry
    from src.core.container import TimeSeriesDataContainer

    # Import models to ensure registration
    import src.models  # noqa: F401

    # Validate model exists
    if not ModelRegistry.is_registered(model_name):
        raise ValueError(
            f"Unknown model '{model_name}'. Available: {ModelRegistry.list_all()}"
        )

    # Get model info
    model_info = ModelRegistry.get_model_info(model_name)
    is_sequence = model_info["requires_sequences"]

    # Load data
    logger.info(f"Loading data from {data_dir}...")
    container = TimeSeriesDataContainer.from_parquet_dir(
        path=data_dir,
        horizon=horizon,
    )
    logger.info(f"Loaded container: {container}")

    # Get arrays (convert to numpy to satisfy type checker)
    X_train_raw, y_train_raw, w_train = container.get_sklearn_arrays("train")
    X_val_raw, y_val_raw, w_val = container.get_sklearn_arrays("val")
    X_train = np.asarray(X_train_raw)
    y_train = np.asarray(y_train_raw)
    X_val = np.asarray(X_val_raw)
    y_val = np.asarray(y_val_raw)

    # Get DataFrames for time validation
    train_df = container.splits["train"].df
    val_df = container.splits["val"].df
    test_df = container.splits.get("test", None)
    if test_df is not None:
        test_df = test_df.df

    # Get test arrays if available
    y_test: np.ndarray | None = None
    if "test" in container.splits:
        _, y_test_raw, _ = container.get_sklearn_arrays("test")
        y_test = np.asarray(y_test_raw)

    # Initialize results
    results = {
        "model_name": model_name,
        "horizon": horizon,
        "data_dir": str(data_dir),
        "model_info": {
            "family": model_info["family"],
            "requires_sequences": is_sequence,
        },
        "data_shape": {
            "n_train": len(X_train),
            "n_val": len(X_val),
            "n_features": X_train.shape[1] if len(X_train.shape) > 1 else 0,
        },
        "tests": {},
    }

    # Determine which tests to run
    all_tests = ["overfit", "baselines", "labels", "confusion", "time_splits"]
    if tests is None:
        tests = all_tests
    else:
        tests = [t.strip().lower() for t in tests]
        invalid = [t for t in tests if t not in all_tests]
        if invalid:
            logger.warning(f"Unknown tests (skipping): {invalid}")
            tests = [t for t in tests if t in all_tests]

    # Run tests
    if "overfit" in tests:
        logger.info("\n" + "=" * 70)
        logger.info("1. OVERFIT TEST")
        logger.info("=" * 70)
        results["tests"]["overfit"] = run_overfit_test(
            model_name=model_name,
            X_train=X_train,
            y_train=y_train,
            n_samples=min(OVERFIT_SAMPLES, len(X_train)),
            is_sequence=is_sequence,
            seq_len=seq_len,
        )

    if "baselines" in tests:
        logger.info("\n" + "=" * 70)
        logger.info("2. BASELINE COMPARISONS")
        logger.info("=" * 70)
        results["tests"]["baselines"] = compute_baseline_metrics(y_val)

    if "labels" in tests:
        logger.info("\n" + "=" * 70)
        logger.info("3. LABEL DISTRIBUTION ANALYSIS")
        logger.info("=" * 70)
        results["tests"]["label_distribution"] = analyze_label_distribution(
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
        )

    if "confusion" in tests:
        logger.info("\n" + "=" * 70)
        logger.info("4. CONFUSION MATRIX ANALYSIS")
        logger.info("=" * 70)

        # Train a model quickly to get predictions
        logger.info("Training model for confusion analysis...")
        model = ModelRegistry.create(model_name)

        # For sequence models, create sequences
        X_train_model = X_train
        X_val_model = X_val
        y_train_model = y_train
        y_val_model = y_val

        if is_sequence:
            try:
                # Create sequences from 2D data for sequence models
                X_train_model, y_train_model = create_sequences_from_arrays(
                    X_train, y_train, seq_len=seq_len
                )
                X_val_model, y_val_model = create_sequences_from_arrays(
                    X_val, y_val, seq_len=seq_len
                )
            except Exception as e:
                logger.warning(f"Could not create sequences: {e}. Using 2D data.")

        try:
            model.fit(
                X_train=X_train_model,
                y_train=y_train_model,
                X_val=X_val_model,
                y_val=y_val_model,
            )
            predictions = model.predict(X_val_model)

            results["tests"]["confusion_matrix"] = analyze_confusion_matrix(
                y_true=y_val_model,
                y_pred=predictions.class_predictions,
                y_proba=predictions.class_probabilities,
            )
        except Exception as e:
            logger.error(f"Confusion analysis failed: {e}")
            results["tests"]["confusion_matrix"] = {"error": str(e)}

    if "time_splits" in tests:
        logger.info("\n" + "=" * 70)
        logger.info("5. TIME SPLIT VALIDATION")
        logger.info("=" * 70)
        results["tests"]["time_splits"] = validate_time_splits(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
        )

    return results


def print_diagnostic_report(results: dict[str, Any]) -> None:
    """Print a formatted diagnostic report."""
    print("\n" + "=" * 80)
    print("MODEL PERFORMANCE DIAGNOSTIC REPORT")
    print("=" * 80)

    print(f"\nModel: {results['model_name']}")
    print(f"Horizon: H{results['horizon']}")
    print(f"Model Family: {results['model_info']['family']}")
    print(f"Requires Sequences: {results['model_info']['requires_sequences']}")
    print(f"\nData Shape:")
    print(f"  Train samples: {results['data_shape']['n_train']:,}")
    print(f"  Val samples: {results['data_shape']['n_val']:,}")
    print(f"  Features: {results['data_shape']['n_features']}")

    tests = results.get("tests", {})

    # 1. Overfit Test
    if "overfit" in tests:
        print("\n" + "-" * 80)
        print("1. OVERFIT TEST")
        print("-" * 80)
        overfit = tests["overfit"]
        if "error" in overfit:
            print(f"  ERROR: {overfit['error']}")
        else:
            status = "PASSED" if overfit["passed"] else "FAILED"
            print(f"  Status: {status}")
            print(f"  Train Accuracy: {overfit['train_accuracy']:.4f}")
            print(f"  Train F1: {overfit['train_f1']:.4f}")
            print(f"  Train Loss: {overfit['train_loss']:.4f}")
            print(f"  Val Loss: {overfit['val_loss']:.4f}")
            print(f"  Epochs: {overfit['epochs_trained']}")
            print(f"  Samples used: {overfit['n_samples']}")
            if not overfit["passed"]:
                print("\n  ISSUE: Model cannot overfit small data.")
                print("  Possible causes:")
                print("    - Labels are random/noisy (no signal)")
                print("    - Model architecture bug")
                print("    - Data pipeline issue")
                print("    - Feature engineering produces useless features")

    # 2. Baseline Comparisons
    if "baselines" in tests:
        print("\n" + "-" * 80)
        print("2. BASELINE COMPARISONS")
        print("-" * 80)
        print(f"  {'Baseline':<20} {'Accuracy':>10} {'F1 Macro':>10}")
        print("  " + "-" * 42)
        for name, metrics in tests["baselines"].items():
            print(
                f"  {name:<20} {metrics['accuracy']:>10.4f} {metrics['f1_macro']:>10.4f}"
            )
        print("\n  Model should significantly beat these baselines.")
        print(f"  Coin-flip (3-class) accuracy: ~{COIN_FLIP_ACCURACY:.3f}")

    # 3. Label Distribution
    if "label_distribution" in tests:
        print("\n" + "-" * 80)
        print("3. LABEL DISTRIBUTION ANALYSIS")
        print("-" * 80)
        dist = tests["label_distribution"]

        for split_name, split_data in dist["splits"].items():
            print(f"\n  {split_name.upper()} ({split_data['n_samples']:,} samples):")
            for label, pct in split_data["percentages"].items():
                count = split_data["distribution"][label]
                print(f"    {label:<10}: {count:>8,} ({pct:>5.1f}%)")

        print(f"\n  Class Balance:")
        balance = dist["class_balance"]
        print(f"    Imbalance score: {balance['imbalance_score']:.4f}")
        print(f"    Is balanced: {balance['is_balanced']}")

        print(f"\n  Autocorrelation (label persistence):")
        for split_name, ac in dist["autocorrelation"].items():
            print(f"    {split_name}: {ac['lag1_persistence']:.4f}")
        print("    (High persistence = labels are sticky, model may just predict previous)")

    # 4. Confusion Matrix
    if "confusion_matrix" in tests:
        print("\n" + "-" * 80)
        print("4. CONFUSION MATRIX ANALYSIS")
        print("-" * 80)
        cm_data = tests["confusion_matrix"]
        if "error" in cm_data:
            print(f"  ERROR: {cm_data['error']}")
        else:
            print("\n  Confusion Matrix:")
            print("            Predicted")
            print(f"            {'short':>8} {'neutral':>8} {'long':>8}")
            print("  Actual")
            for i, label in enumerate(["short", "neutral", "long"]):
                row = cm_data["confusion_matrix"][i]
                print(f"    {label:<8} {row[0]:>8} {row[1]:>8} {row[2]:>8}")

            print("\n  Per-Class Metrics:")
            print(f"  {'Class':<10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
            print("  " + "-" * 52)
            for cls, metrics in cm_data["per_class"].items():
                print(
                    f"  {cls:<10} {metrics['precision']:>10.4f} {metrics['recall']:>10.4f} "
                    f"{metrics['f1']:>10.4f} {metrics['support']:>10}"
                )

            print(f"\n  Overall:")
            print(f"    Accuracy: {cm_data['overall']['accuracy']:.4f}")
            print(f"    Macro F1: {cm_data['overall']['macro_f1']:.4f}")
            print(f"    Weighted F1: {cm_data['overall']['weighted_f1']:.4f}")

            if "threshold_analysis" in cm_data:
                print("\n  Threshold Sensitivity:")
                print(f"  {'Threshold':>10} {'Accuracy':>10} {'F1':>10} {'Coverage':>10}")
                print("  " + "-" * 42)
                for t in cm_data["threshold_analysis"]:
                    print(
                        f"  {t['threshold']:>10.2f} {t['accuracy']:>10.4f} "
                        f"{t['f1_macro']:>10.4f} {t['coverage']:>10.4f}"
                    )

    # 5. Time Split Validation
    if "time_splits" in tests:
        print("\n" + "-" * 80)
        print("5. TIME SPLIT VALIDATION")
        print("-" * 80)
        ts = tests["time_splits"]
        print(f"  Ordering Valid: {ts['ordering_valid']}")
        print(f"  Leakage Detected: {ts['leakage_detected']}")

        for split_name, split_data in ts["splits"].items():
            print(f"\n  {split_name.upper()}:")
            print(f"    Start: {split_data['start']}")
            print(f"    End: {split_data['end']}")
            print(f"    Samples: {split_data['n_samples']:,}")

        if ts["issues"]:
            print("\n  ISSUES:")
            for issue in ts["issues"]:
                print(f"    - {issue}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    issues = []
    if "overfit" in tests and not tests["overfit"].get("passed", False):
        issues.append("Model cannot overfit small data - check pipeline/model")

    if "label_distribution" in tests:
        dist = tests["label_distribution"]
        if not dist["class_balance"]["is_balanced"]:
            issues.append("Class imbalance detected - consider rebalancing")
        for split_name, ac in dist["autocorrelation"].items():
            if ac["lag1_persistence"] > 0.7:
                issues.append(f"High label persistence in {split_name} - labels may be too sticky")

    if "time_splits" in tests and tests["time_splits"]["leakage_detected"]:
        issues.append("CRITICAL: Data leakage detected in time splits!")

    if issues:
        print("\nPotential Issues Found:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
    else:
        print("\nNo obvious issues detected.")

    print("\n" + "=" * 80)


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Diagnose model performance issues",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick diagnostic with XGBoost
  python scripts/diagnose_model_performance.py --model xgboost --horizon 20

  # Full diagnostic with LSTM
  python scripts/diagnose_model_performance.py --model lstm --horizon 20 --seq-len 30

  # Run only specific tests
  python scripts/diagnose_model_performance.py --model xgboost --horizon 20 \
      --tests overfit,baselines

  # List available models
  python scripts/diagnose_model_performance.py --list-models
        """,
    )

    parser.add_argument(
        "--model",
        type=str,
        help="Model name to diagnose (e.g., xgboost, lstm, tcn)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=20,
        help="Label horizon (default: 20)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Path to scaled data directory (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=30,
        help="Sequence length for sequence models (default: 30)",
    )
    parser.add_argument(
        "--tests",
        type=str,
        help="Comma-separated tests to run: overfit,baselines,labels,confusion,time_splits",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Path to save JSON results",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models and exit",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()
    setup_logging(args.verbose)

    # Handle list-models
    if args.list_models:
        from src.models.registry import ModelRegistry

        import src.models  # noqa: F401

        print("\nAvailable Models:")
        for family, models in sorted(ModelRegistry.list_models().items()):
            print(f"\n  {family.upper()}:")
            for model in sorted(models):
                print(f"    - {model}")
        print(f"\nTotal: {ModelRegistry.count()} models")
        return 0

    # Validate required args
    if not args.model:
        print("Error: --model is required")
        print("Use --list-models to see available models")
        return 1

    # Validate data directory
    if not args.data_dir.exists():
        print(f"Error: Data directory not found: {args.data_dir}")
        print("Run Phase 1 pipeline first to generate scaled data")
        return 1

    # Parse tests
    tests = None
    if args.tests:
        tests = args.tests.split(",")

    # Run diagnostics
    try:
        results = run_diagnostics(
            model_name=args.model,
            horizon=args.horizon,
            data_dir=args.data_dir,
            seq_len=args.seq_len,
            tests=tests,
        )
    except Exception as e:
        logger.exception(f"Diagnostics failed: {e}")
        return 1

    # Print report
    print_diagnostic_report(results)

    # Save JSON if requested
    if args.output:
        import json

        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
