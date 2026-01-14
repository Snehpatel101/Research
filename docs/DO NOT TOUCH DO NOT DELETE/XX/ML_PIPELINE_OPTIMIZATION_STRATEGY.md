# ML Pipeline Optimization Strategy for OHLCV Time Series

## Executive Summary

This document outlines a comprehensive strategy for upgrading the existing machine learning pipeline from a research-oriented Jupyter notebook into a robust, production-ready system for financial time series forecasting. The primary objectives are **reproducibility**, **leakage prevention**, and **financial alignment**. By transitioning from ad-hoc experimentation to a structured MLOps workflow, we aim to increase the reliability of our trading signals and accelerate the iteration cycle for new strategies.

## 1. Data Engineering & Integrity

Financial data is prone to silent errors that can destroy model performance. A rigid data engineering layer is the foundation of the pipeline.

### 1.1 Automated Data Validation
Before any training occurs, data must pass a strict validation suite. This should be implemented as a standalone module (e.g., `src/data/validation.py`) invoked by the pipeline.
*   **Timestamp Integrity:** Verify strictly monotonic increasing timestamps with no duplicates. Detect missing bars (gaps) during active market hours.
*   **OHLCV Logic:** Enforce invariants: `Low <= Open, Close <= High` and `Volume >= 0`.
*   **Outlier Detection:** Flag or clip tick data spikes that exceed $N$ standard deviations from a rolling mean, as these often represent bad data points rather than genuine market moves.

### 1.2 Session Management
Explicitly handle trading sessions. Models should not "learn" overnight gaps as standard volatility.
*   **Filtering:** Option to filter out extended hours if the strategy focuses on liquid market sessions.
*   **Session Boundaries:** When calculating rolling features (e.g., VWAP), ensure counters reset at the start of each session if required by the indicator logic.

## 2. Advanced Feature Engineering

The current pipeline likely relies on standard indicators. The optimization strategy involves creating a feature generation engine that emphasizes stationarity and regime awareness.

### 2.1 Stationarity & Fractional Differentiation
Financial time series are non-stationary (statistical properties change over time). Standard differencing removes memory, while raw prices preserve memory but drift.
*   **Strategy:** Implement **Fractional Differencing** (per López de Prado) to achieve stationarity while preserving the maximum amount of memory (correlation with past data).
*   **Log-Returns:** Use log-returns for price inputs to ensure scale invariance.

### 2.2 Regime Detection
Market dynamics shift between trending, mean-reverting, and volatile regimes. A single model often struggles to master all three.
*   **Implementation:** Integrate the `add_regime_detection.py` logic directly into the preprocessing pipeline.
*   **Usage:** Use detected regimes either as an explicit categorical feature or to train separate "regime-specialist" models (Mixture of Experts approach).

### 2.3 Feature Selection & Orthogonality
A common pitfall is feeding highly correlated indicators (e.g., EMA_20 and EMA_50) which degrades tree-based model performance via feature substitution.
*   **Clustering:** Apply Hierarchical Risk Parity (HRP) or simple correlation clustering to group similar features and select one representative per cluster.
*   **Importance:** Use Permutation Feature Importance (PFI) on a hold-out set, rather than impurity-based importance, to identify truly predictive signals.

## 3. Rigorous Validation Framework

Random k-fold cross-validation is mathematically invalid for time series due to autocorrelation. We must adopt a validation scheme that mimics the deployment reality.

### 3.1 Embargoed Combinatorial CV (PurgedKFold)
To maximize data usage while preventing leakage:
*   **Purging:** Remove samples from the training set that overlap in time with the test set labels (crucial for long-horizon targets).
*   **Embargo:** Drop a buffer period *after* each test set to prevent "bleeding" of information from the test set back into the training set via long-memory features.

### 3.2 Walk-Forward Optimization (WFO)
For final model verification, use an expanding window approach:
1.  Train on $[t_0, t_k]$.
2.  Test on $[t_k, t_{k+1}]$.
3.  Expand training to $[t_0, t_{k+1}]$ and repeat.
This provides the most realistic estimate of how the model would have performed historically.

## 4. Financial Objective Functions

Standard ML loss functions (MSE, Cross-Entropy) optimize for statistical fit, not trading profit. The pipeline must optimize for **risk-adjusted returns**.

### 4.1 Custom Loss Functions
*   **Directional Accuracy penalty:** Penalize errors heavily if the predicted sign is wrong, even if the magnitude error is small.
*   **Weighted Loss:** Weight samples by volatility or volume. High-volatility periods represent higher risk/reward; the model should prioritize accuracy during these times.

### 4.2 Trading-Aware Metrics
Model evaluation should report:
*   **Sharpe/Sortino Ratio:** Risk-adjusted return.
*   **Max Drawdown:** The worst peak-to-trough decline.
*   **Profit Factor:** Gross Profit / Gross Loss.
*   **Win Rate:** Percentage of profitable trades.
*   **Edge:** Average profit per trade.

## 5. MLOps & Architecture Refactoring

To move from "notebook" to "product," we must decouple orchestration from implementation.

### 5.1 Modularization
Refactor `ML_Pipeline.ipynb` into a package structure:
*   `src/features/`: Transformations and indicator logic.
*   `src/models/`: Model definitions (XGBoost, LSTM, etc.).
*   `src/training/`: Training loops and cross-validation logic.
*   `src/evaluation/`: Backtesting and metric calculation.

The notebook then becomes a lightweight controller that imports these modules, runs the pipeline, and visualizes results.

### 5.2 Configuration Management
Hardcoded paths and hyperparameters must be removed.
*   **Hydra / YAML:** Use a configuration file (e.g., `config/experiment_001.yaml`) to define feature sets, model params, and date ranges. This ensures every experiment is documented and reproducible.

### 5.3 Artifact Tracking
Use `MLflow` or a simple JSON/SQLite based tracker to log:
*   Git commit hash.
*   Config parameters used.
*   Final validation metrics.
*   Path to the saved model binary.

## 6. Implementation Roadmap

1.  **Phase 1: Stabilization (Days 1-2)**
    *   Implement `src/paths.py` to unify `PROJECT_ROOT` handling.
    *   Create `src/data/validation.py` and add the data audit cell to the notebook.
    *   Externalize all hyperparameters to a `config.yaml`.

2.  **Phase 2: Validation Upgrade (Days 3-5)**
    *   Replace random splits with `PurgedWalkForwardCV`.
    *   Implement the "Trading Metrics" evaluation block to replace simple Accuracy/F1 scores.

3.  **Phase 3: Feature & Model Refinement (Days 6-10)**
    *   Integrate Regime Detection.
    *   Implement Optuna tuning minimizing `Negative Sharpe Ratio` instead of `LogLoss`.

4.  **Phase 4: Full Automation (Day 10+)**
    *   Convert the notebook into a CLI script (`train.py`) that can be run in the background.
    *   Set up a "Leaderboard" to track the best performing models across experiments.

## Conclusion

By adopting this strategy, we shift the focus from "training a model" to "building a trading system." The added rigor in validation and data engineering will likely reduce initial reported performance (as leakage is removed) but will significantly increase the correlation between backtest results and live trading performance.
