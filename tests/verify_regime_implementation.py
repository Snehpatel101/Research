#!/usr/bin/env python3
"""
Verify regime detection implementation in ML_Pipeline.ipynb.

This script validates that all regime detection features were correctly added.
"""

import json
from pathlib import Path


def verify_notebook():
    """Verify all regime detection changes."""
    notebook_path = Path('/home/jake/Desktop/Research/notebooks/ML_Pipeline.ipynb')

    with open(notebook_path) as f:
        notebook = json.load(f)

    print("=" * 70)
    print(" REGIME DETECTION IMPLEMENTATION VERIFICATION")
    print("=" * 70)

    checks = []

    # Check 1: Notebook has correct number of cells
    n_cells = len(notebook['cells'])
    checks.append(("Notebook has 28 cells", n_cells == 28, f"Found {n_cells} cells"))

    # Check 2: Cell 4 has regime fields
    cell_4_source = '\n'.join(notebook['cells'][4]['source'])
    has_regime_labels = 'regime_labels: Optional[np.ndarray]' in cell_4_source
    checks.append(("NotebookConfig has regime_labels field", has_regime_labels, ""))

    has_regime_probs = 'regime_probabilities: Optional[np.ndarray]' in cell_4_source
    checks.append(("NotebookConfig has regime_probabilities field", has_regime_probs, ""))

    has_n_regimes = 'n_regimes: int = 4' in cell_4_source
    checks.append(("NotebookConfig has n_regimes field", has_n_regimes, ""))

    # Check 3: Cell 11 is regime detection cell
    cell_11_source = '\n'.join(notebook['cells'][11]['source'])
    is_regime_cell = '3.4 Detect Market Regimes' in cell_11_source
    checks.append(("Cell 11 is '3.4 Detect Market Regimes'", is_regime_cell, ""))

    has_hmm_import = 'from src.phase1.stages.regime import HMMRegimeDetector' in cell_11_source
    checks.append(("Cell 11 imports HMMRegimeDetector", has_hmm_import, ""))

    has_hmm_init = 'HMMRegimeDetector(' in cell_11_source
    checks.append(("Cell 11 initializes HMMRegimeDetector", has_hmm_init, ""))

    stores_regime_labels = 'config.regime_labels = regimes.values' in cell_11_source
    checks.append(("Cell 11 stores regime_labels", stores_regime_labels, ""))

    # Check 4: Cell 14 stores val_predictions
    cell_14_source = '\n'.join(notebook['cells'][14]['source'])
    stores_val_preds = "'val_predictions': result.val_predictions" in cell_14_source
    checks.append(("Cell 14 stores val_predictions", stores_val_preds, ""))

    # Check 5: Cell 15 has per-regime performance
    cell_15_source = '\n'.join(notebook['cells'][15]['source'])
    has_regime_perf = 'PER-REGIME PERFORMANCE BREAKDOWN' in cell_15_source
    checks.append(("Cell 15 has per-regime performance section", has_regime_perf, ""))

    has_regime_check = "config.regime_labels is not None" in cell_15_source
    checks.append(("Cell 15 checks for regime_labels", has_regime_check, ""))

    # Check 6: Cell 16 is regime heatmap
    cell_16_source = '\n'.join(notebook['cells'][16]['source'])
    is_heatmap_cell = '4.3 Regime Performance Heatmap' in cell_16_source
    checks.append(("Cell 16 is '4.3 Regime Performance Heatmap'", is_heatmap_cell, ""))

    has_heatmap = 'sns.heatmap(' in cell_16_source
    checks.append(("Cell 16 creates heatmap", has_heatmap, ""))

    has_regime_analysis = 'Most Regime-Agnostic' in cell_16_source
    checks.append(("Cell 16 has regime analysis", has_regime_analysis, ""))

    # Print results
    print("\nVerification Results:")
    print("-" * 70)

    all_passed = True
    for check_name, passed, detail in checks:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:10s} {check_name}")
        if detail:
            print(f"           {detail}")
        if not passed:
            all_passed = False

    print("-" * 70)
    print(f"\nTotal Checks: {len(checks)}")
    print(f"Passed: {sum(1 for _, p, _ in checks if p)}")
    print(f"Failed: {sum(1 for _, p, _ in checks if not p)}")

    if all_passed:
        print("\n✓ ALL CHECKS PASSED - Implementation verified!")
    else:
        print("\n✗ SOME CHECKS FAILED - Review errors above")

    print("=" * 70)

    return all_passed


if __name__ == '__main__':
    success = verify_notebook()
    exit(0 if success else 1)
