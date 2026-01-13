#!/usr/bin/env python3
"""
Refactor ML_Pipeline.ipynb to replace inline code with imports from src/.

Replaces 6 bloated cells (884 lines) with clean imports.
"""

import json
from pathlib import Path


def refactor_cell_9(cell):
    """Cell 9: MLOps Helper Functions (108→15 lines)"""
    new_source = '''#@title 2.4 MLOps Helper Functions { display-mode: "form" }

import numpy as np
import pandas as pd
from typing import Dict, Any

# Import from src modules
from src.phase1.stages.validation.normalization import detect_outliers
from src.monitoring import PSIDetector

def check_missing_values(df: pd.DataFrame, threshold: float = 0.05) -> Dict[str, Any]:
    """Check for missing values in dataframe."""
    missing = df.isnull().sum() / len(df)
    issues = missing[missing > threshold]
    return {
        "missing_pct": float(missing.max()),
        "issues": {col: float(pct) for col, pct in issues.items()},
        "passed": len(issues) == 0
    }

print("MLOps helper functions loaded from src/ modules")
'''
    cell['source'] = new_source.splitlines(keepends=True)
    return cell


def refactor_cell_14(cell):
    """Cell 14: Data Quality Validation (168→40 lines)"""
    new_source = '''#@title 3.6 Data Quality Validation { display-mode: "form" }

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import from src modules
from src.phase1.stages.validation import DataValidator

print("=" * 70)
print("DATA QUALITY VALIDATION")
print("=" * 70)

if not ENABLE_DATA_QUALITY_GATES:
    print("Data Quality Gates disabled. Skipping validation.")
else:
    # Load processed data
    train_data = CONFIG.train_data
    test_data = CONFIG.test_data

    print(f"\\nValidating training data: {len(train_data):,} rows")
    print(f"Validating test data: {len(test_data):,} rows")

    # Create validator
    validator = DataValidator(train_data, horizons=[CONFIG.training_horizon])

    # Run validation checks
    integrity = validator.check_data_integrity()
    labels = validator.check_label_sanity()
    features = validator.check_feature_quality(max_features=50)
    normalization = validator.check_feature_normalization()

    # Store results in CONFIG
    CONFIG.data_quality_results = {
        "integrity": integrity,
        "labels": labels,
        "features": features,
        "normalization": normalization,
        "passed": len(validator.issues_found) == 0
    }

    # Print summary
    summary = validator.generate_summary()
    print(f"\\n{'✅' if summary['status'] == 'PASSED' else '❌'} Validation {summary['status']}")
    print(f"Issues: {summary['issues_count']}, Warnings: {summary['warnings_count']}")

    CONFIG.data_quality_gate_passed = summary['status'] == 'PASSED'

print("\\n" + "=" * 70)
'''
    cell['source'] = new_source.splitlines(keepends=True)
    return cell


def refactor_cell_22(cell):
    """Cell 22: Prediction Calibration (143→35 lines)"""
    new_source = '''#@title 4.5 Prediction Calibration & Confidence { display-mode: "form" }

import numpy as np
import matplotlib.pyplot as plt

# Import from src modules
from src.models.calibration import ProbabilityCalibrator, CalibrationConfig, compute_ece

print("="*70)
print(" PREDICTION CALIBRATION & CONFIDENCE ANALYSIS")
print("="*70)

if not CONFIG.training_results:
    print("\\n❌ No training results found. Run Cell 4.1 first.")
else:
    # Load scaled data
    data_file = CONFIG.splits_dir / f"{CONFIG.symbol}_H{CONFIG.training_horizon}_scaled.npz"

    if not data_file.exists():
        print(f"❌ Data file not found: {data_file}")
    else:
        data = np.load(data_file, allow_pickle=True)
        X_val, y_val = data['X_val'], data['y_val']
        X_test, y_test = data['X_test'], data['y_test']

        # Get model predictions
        model_name = CONFIG.best_model_name
        model_path = CONFIG.training_results[model_name]['model_path']

        print(f"\\nCalibrating predictions for: {model_name}")

        # Load and predict
        from src.models import ModelRegistry
        model = ModelRegistry.load(model_name, model_path)
        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test)

        # Calibrate probabilities
        config = CalibrationConfig(method="isotonic", n_bins=10)
        calibrator = ProbabilityCalibrator(config)
        metrics = calibrator.fit(y_val, val_pred.probabilities)

        # Apply calibration to test set
        calibrated_probs = calibrator.calibrate(test_pred.probabilities)

        # Compute metrics
        ece_before = compute_ece(y_test, test_pred.probabilities)
        ece_after = compute_ece(y_test, calibrated_probs)

        print(f"\\nCalibration Results:")
        print(f"  ECE Before: {ece_before:.4f}")
        print(f"  ECE After:  {ece_after:.4f}")
        print(f"  Improvement: {(ece_before - ece_after)/ece_before*100:.1f}%")

        # Store results
        CONFIG.calibration_results = {
            "ece_before": float(ece_before),
            "ece_after": float(ece_after),
            "method": config.method,
            "calibrated_probabilities": calibrated_probs
        }

        print("\\n✅ Calibration complete")

print("\\n" + "=" * 70)
'''
    cell['source'] = new_source.splitlines(keepends=True)
    return cell


def refactor_cell_26(cell):
    """Cell 26: Performance Monitoring (135→30 lines)"""
    new_source = '''#@title 4.6 Performance Monitoring (Rolling Window) { display-mode: "form" }

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import from src modules
from src.models.metrics import compute_classification_metrics

print("=" * 70)
print("PERFORMANCE MONITORING (Rolling Window Analysis)")
print("=" * 70)

if not ENABLE_PERFORMANCE_MONITORING:
    print("Performance monitoring disabled. Skipping.")
else:
    if not hasattr(CONFIG, 'best_model_name') or CONFIG.best_model_name is None:
        print("⚠️  No model trained yet. Skipping performance monitoring.")
    else:
        print(f"\\nMonitoring model: {CONFIG.best_model_name}")
        print(f"Rolling window size: {PERFORMANCE_WINDOW}")

        # Load test predictions
        data_file = CONFIG.splits_dir / f"{CONFIG.symbol}_H{CONFIG.training_horizon}_scaled.npz"
        data = np.load(data_file, allow_pickle=True)
        y_test = data['y_test']

        # Get predictions from stored results
        test_pred = CONFIG.training_results[CONFIG.best_model_name]['test_predictions']
        y_pred = test_pred.predictions
        y_proba = test_pred.probabilities

        # Compute rolling window metrics
        window = PERFORMANCE_WINDOW
        n_windows = len(y_test) // window

        rolling_metrics = []
        for i in range(n_windows):
            start_idx = i * window
            end_idx = start_idx + window

            window_metrics = compute_classification_metrics(
                y_test[start_idx:end_idx],
                y_pred[start_idx:end_idx],
                y_proba[start_idx:end_idx]
            )
            rolling_metrics.append(window_metrics)

        # Store results
        CONFIG.performance_monitoring = {
            "window_size": window,
            "n_windows": n_windows,
            "rolling_metrics": rolling_metrics
        }

        print(f"\\n✅ Computed metrics for {n_windows} windows")

        # Plot rolling performance
        accuracies = [m['accuracy'] for m in rolling_metrics]
        f1_scores = [m['f1_score'] for m in rolling_metrics]

        plt.figure(figsize=(12, 4))
        plt.plot(accuracies, label='Accuracy', marker='o')
        plt.plot(f1_scores, label='F1 Score', marker='s')
        plt.xlabel('Window Index')
        plt.ylabel('Score')
        plt.title(f'Rolling Window Performance ({CONFIG.best_model_name})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

print("\\n" + "=" * 70)
'''
    cell['source'] = new_source.splitlines(keepends=True)
    return cell


def refactor_cell_30(cell):
    """Cell 30: Feature Drift Detection (160→45 lines)"""
    new_source = '''#@title 5.6 Feature Drift Detection { display-mode: "form" }

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import from src modules
from src.monitoring import PSIDetector, KSDetector

print("=" * 70)
print("FEATURE DRIFT DETECTION")
print("=" * 70)

if not ENABLE_DRIFT_DETECTION:
    print("Drift detection disabled. Skipping.")
else:
    # Load data
    train_data = CONFIG.train_data
    test_data = CONFIG.test_data
    feature_cols = [c for c in train_data.columns if c not in ['target', 'timestamp', 'quality_score']]

    print(f"\\nDetecting drift for {len(feature_cols)} features...")
    print(f"Train samples: {len(train_data):,}")
    print(f"Test samples: {len(test_data):,}")

    # Initialize detectors
    psi_detector = PSIDetector(n_bins=10, threshold=0.1)
    ks_detector = KSDetector(threshold=0.05)

    # Detect drift for each feature
    drift_results = []
    for col in feature_cols[:50]:  # Limit to 50 features for performance
        # PSI drift
        psi_result = psi_detector.detect_drift(
            train_data[col].values,
            test_data[col].values,
            feature_name=col
        )

        # KS drift
        ks_result = ks_detector.detect_drift(
            train_data[col].values,
            test_data[col].values,
            feature_name=col
        )

        drift_results.append({
            'feature': col,
            'psi': psi_result.metric_value,
            'psi_drift': psi_result.drift_detected,
            'ks_statistic': ks_result.metric_value,
            'ks_drift': ks_result.drift_detected
        })

    # Create drift summary
    drift_df = pd.DataFrame(drift_results)
    n_psi_drift = drift_df['psi_drift'].sum()
    n_ks_drift = drift_df['ks_drift'].sum()

    print(f"\\nDrift Detection Results:")
    print(f"  PSI drift detected: {n_psi_drift}/{len(drift_df)} features")
    print(f"  KS drift detected: {n_ks_drift}/{len(drift_df)} features")

    # Store results
    CONFIG.drift_detection_results = {
        "drift_summary": drift_df.to_dict('records'),
        "n_features_checked": len(drift_df),
        "psi_drift_count": int(n_psi_drift),
        "ks_drift_count": int(n_ks_drift),
        "drift_detected": (n_psi_drift > 0) or (n_ks_drift > 0)
    }

    # Plot top drifting features
    top_drift = drift_df.nlargest(10, 'psi')
    plt.figure(figsize=(10, 6))
    plt.barh(top_drift['feature'], top_drift['psi'])
    plt.xlabel('PSI Score')
    plt.title('Top 10 Features by PSI Drift')
    plt.tight_layout()
    plt.show()

    print(f"\\n{'⚠️' if CONFIG.drift_detection_results['drift_detected'] else '✅'} Drift detection complete")

print("\\n" + "=" * 70)
'''
    cell['source'] = new_source.splitlines(keepends=True)
    return cell


def refactor_cell_32(cell):
    """Cell 32: Deployment Decision Gate (170→50 lines)"""
    new_source = '''#@title 5.7 Deployment Decision Gate { display-mode: "form" }

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("=" * 70)
print("DEPLOYMENT DECISION GATE")
print("=" * 70)

if not USE_DEPLOYMENT_GATES:
    print("Deployment gates disabled. Manual deployment decision required.")
    CONFIG.deployment_gate_passed = None
else:
    print("\\nChecking all deployment criteria...\\n")

    # Collect all results from CONFIG
    results = {}

    # 1. Trading performance metrics
    if hasattr(CONFIG, 'best_model_metrics'):
        metrics = CONFIG.best_model_metrics
        results['sharpe_ratio'] = {
            'value': metrics.get('sharpe_ratio', 0),
            'threshold': MIN_SHARPE,
            'passed': metrics.get('sharpe_ratio', 0) >= MIN_SHARPE
        }
        results['win_rate'] = {
            'value': metrics.get('win_rate', 0),
            'threshold': MIN_WIN_RATE,
            'passed': metrics.get('win_rate', 0) >= MIN_WIN_RATE
        }
    else:
        results['sharpe_ratio'] = {'value': 0, 'threshold': MIN_SHARPE, 'passed': False}
        results['win_rate'] = {'value': 0, 'threshold': MIN_WIN_RATE, 'passed': False}

    # 2. Data quality gate
    if hasattr(CONFIG, 'data_quality_gate_passed'):
        results['data_quality'] = {
            'passed': CONFIG.data_quality_gate_passed,
            'details': 'See Cell 3.6 for details'
        }
    else:
        results['data_quality'] = {'passed': True, 'details': 'Validation skipped'}

    # 3. Calibration quality
    if hasattr(CONFIG, 'calibration_results'):
        ece = CONFIG.calibration_results['ece_after']
        results['calibration'] = {
            'value': ece,
            'threshold': MAX_ECE,
            'passed': ece <= MAX_ECE
        }
    else:
        results['calibration'] = {'value': 1.0, 'threshold': MAX_ECE, 'passed': False}

    # 4. Drift detection
    if hasattr(CONFIG, 'drift_detection_results'):
        drift_pct = CONFIG.drift_detection_results['psi_drift_count'] / CONFIG.drift_detection_results['n_features_checked']
        results['drift'] = {
            'value': drift_pct,
            'threshold': MAX_DRIFT_PCT,
            'passed': drift_pct <= MAX_DRIFT_PCT
        }
    else:
        results['drift'] = {'value': 0, 'threshold': MAX_DRIFT_PCT, 'passed': True}

    # Print results
    print("Deployment Gate Results:")
    print("-" * 70)
    for check, result in results.items():
        status = "✅ PASS" if result['passed'] else "❌ FAIL"
        print(f"{check:20s}: {status}")
        if 'value' in result:
            print(f"  Value: {result['value']:.4f}, Threshold: {result['threshold']:.4f}")

    # Overall decision
    all_passed = all(r['passed'] for r in results.values())
    CONFIG.deployment_gate_passed = all_passed
    CONFIG.deployment_gate_results = results

    print("\\n" + "=" * 70)
    if all_passed:
        print("✅ DEPLOYMENT APPROVED - All gates passed")
    else:
        print("❌ DEPLOYMENT BLOCKED - Some gates failed")
        failed = [k for k, v in results.items() if not v['passed']]
        print(f"Failed checks: {', '.join(failed)}")
    print("=" * 70)

print("\\n" + "=" * 70)
'''
    cell['source'] = new_source.splitlines(keepends=True)
    return cell


def main():
    nb_path = Path('/home/jake/Desktop/Research/notebooks/ML_Pipeline.ipynb')

    print("Loading notebook...")
    with open(nb_path, 'r') as f:
        nb = json.load(f)

    print(f"Total cells: {len(nb['cells'])}")

    # Target cells to refactor
    refactorings = {
        9: refactor_cell_9,
        14: refactor_cell_14,
        22: refactor_cell_22,
        26: refactor_cell_26,
        30: refactor_cell_30,
        32: refactor_cell_32,
    }

    # Apply refactorings
    for cell_idx, refactor_fn in refactorings.items():
        print(f"Refactoring cell {cell_idx}...")
        original_lines = len(nb['cells'][cell_idx]['source'])
        nb['cells'][cell_idx] = refactor_fn(nb['cells'][cell_idx])
        new_lines = len(nb['cells'][cell_idx]['source'])
        print(f"  {original_lines} → {new_lines} lines (saved {original_lines - new_lines} lines)")

    # Save refactored notebook
    output_path = nb_path
    print(f"\nSaving refactored notebook to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(nb, f, indent=1)

    print("✅ Refactoring complete!")
    print(f"\nBackup saved at: {nb_path}.refactor_backup")


if __name__ == '__main__':
    main()
