# ML Factory - Phase 12 Implementation Tasks

**Status:** PHASE 12 COMPLETE | PHASE 12.5 PENDING
**Completed:** 2026-01-24 (Phase 12)
**Review Date:** 2026-01-25
**Priority:** HIGH - Code quality cleanup before Phase 13

---

## PHASE 12 COMPLETION SUMMARY

**Completed by 7-Agent Pipeline on 2026-01-24**

### Critical Fixes Implemented:
- [x] Optuna now optimizes Sharpe ratio (was F1 - CRITICAL FIX)
- [x] R-multiple tracking for every trade
- [x] Unit tests created (6 test files, 981 lines)
- [x] FeatureStore integration for caching
- [x] MLflow enabled by default
- [x] Circuit breakers for live trading (drawdown, daily loss, consecutive losses)

### All Sub-Phases Complete:
- [x] Phase 12A: Trading Profitability (8 tasks) - COMPLETE
- [x] Phase 12B: Live Trading Safeguards (7 tasks) - COMPLETE
- [x] Phase 12C: Deployment Infrastructure (6 tasks) - COMPLETE (1 skipped)
- [x] Phase 12D: Pipeline Performance (7 tasks) - COMPLETE
- [x] Phase 12E: Testing Infrastructure (5 tasks) - COMPLETE
- [x] Phase 12F: Architecture Cleanup (6 tasks) - COMPLETE (1 skipped)

---

## Completed Phases

See COMPLETION.md for details on Phases 0-11.

---

## Phase 12: Trading Profitability & Live Deployment

**10-Agent Analysis Summary (ALL RESOLVED):**
- [x] Optuna optimizes F1 (WRONG) instead of Sharpe ratio -- FIXED
- [x] No R-multiple tracking (can't measure risk/reward) -- IMPLEMENTED
- [x] Zero unit tests (448 files untested) -- 6 TEST FILES CREATED
- [x] FeatureStore unused (features recomputed every run) -- INTEGRATED
- [x] MLflow not auto-enabled -- NOW DEFAULT
- [x] No circuit breakers for live trading -- IMPLEMENTED

---

## Phase 12A: Trading Profitability (8 tasks)

### 12A-1: P&L-Based Optuna Objective (CRITICAL)

**File:** `/Users/sneh/research/src/optimization/five_dimension_objective.py`

**Current (WRONG):**
```python
# Line 437-444
def default_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="weighted"))
```

**Fix:**
```python
# Replace with Sharpe-based objective
def default_metric_sharpe(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    prices: pd.DataFrame,
    labels: pd.DataFrame,
    costs: TransactionCosts,
) -> float:
    """Optimize for trading profitability, not classification accuracy."""
    from src.inference.backtesting import Backtester, BacktestConfig

    # Quick backtest with predictions
    config = BacktestConfig(
        costs=costs,
        execution_model="market_on_close",
        slippage_model="volatility_scaled",  # Realistic slippage
    )

    backtester = Backtester(config)
    result = backtester.run(prices, predictions=y_pred, labels=y_true)

    # Return Sharpe ratio (or profit factor, Sortino, etc.)
    return result.metrics.sharpe_ratio
```

**Change default at line 879:**
```python
# Line 879: study.optimize(objective, n_trials=100, direction="maximize")
# Update objective to use default_metric_sharpe
```

**Verify:**
```bash
python -m src optimize --metric sharpe_ratio --trials 10 --symbol MES
# Check that optimization log shows Sharpe values, not F1
```

---

### 12A-2: Volatility-Scaled Slippage Default

**File:** `/Users/sneh/research/src/inference/backtesting/costs.py`

**Change line 337:**
```python
# BEFORE:
slippage_model: BaseSlippageModel = field(default_factory=FixedSlippage)

# AFTER:
slippage_model: BaseSlippageModel = field(default_factory=lambda: VolatilityScaledSlippage(
    base_ticks=1.0,
    base_volatility=0.15,
    volatility_multiplier=2.0
))
```

**Verify:**
```bash
python -c "from src.inference.backtesting.costs import CostCalculator; c = CostCalculator(); print(type(c.slippage_model).__name__)"
# Should print: VolatilityScaledSlippage
```

---

### 12A-3: Market Hours Filtering

**File (NEW):** `/Users/sneh/research/src/inference/backtesting/execution.py`

**Create:**
```python
"""
Execution models with realistic constraints.

Includes:
- Market hours filtering (CME calendar)
- Adverse selection bias
- Volume-relative position limits
"""

from datetime import datetime
from src.data.pipeline.stages.sessions import CMECalendar, SessionFilter

class ExecutionModel:
    def is_tradeable_time(self, timestamp: datetime, contract: str = "MES") -> bool:
        """Check if timestamp is during liquid trading hours."""
        calendar = CMECalendar()
        session_filter = SessionFilter()

        # Skip holidays
        if calendar.is_holiday(timestamp):
            return False

        # Only trade during NY session (9:30 AM - 4:00 PM ET)
        session_flags = session_filter.get_session_flags(timestamp)
        return session_flags.get("session_ny", False)

    def apply_adverse_selection(
        self,
        signal_price: float,
        signal_direction: int,
        realized_volatility: float,
    ) -> float:
        """Adjust fill price for adverse selection (market moved before fill)."""
        # Model predicts move → market already moving → worse fill
        adverse_ticks = 0.5 + 0.5 * (realized_volatility / 0.15)
        tick_value = 0.25  # MES

        if signal_direction > 0:  # Long
            return signal_price + adverse_ticks * tick_value
        else:  # Short
            return signal_price - adverse_ticks * tick_value
```

**Integrate into backtest.py:**
```python
# Line 392 in _open_position
from src.inference.backtesting.execution import ExecutionModel

if not self.execution_model.is_tradeable_time(timestamp, contract=self.config.symbol):
    logger.debug(f"Skipping signal at {timestamp}: outside trading hours")
    return  # Don't trade
```

**Verify:**
```bash
python -m src backtest --realistic-execution --symbol MES
# Should skip weekend/holiday trades
```

---

### 12A-4: Volume-Relative Position Limits

**File:** `/Users/sneh/research/src/inference/backtesting/costs.py`

**Add method at line 410:**
```python
def calculate_max_position_size(
    self,
    avg_volume_5min: float,
    max_participation: float = 0.01,
) -> int:
    """
    Limit position size to fraction of market volume.

    Args:
        avg_volume_5min: Rolling 20-bar average volume
        max_participation: Max % of volume (default 1%)

    Returns:
        Maximum contracts tradeable without excessive market impact
    """
    max_contracts = int(avg_volume_5min * max_participation)
    return max(1, max_contracts)  # At least 1 contract
```

**Use in backtest.py line 392:**
```python
# Calculate position size with volume check
requested_contracts = self.position_sizer.calculate_position_size(...)
max_contracts = self.cost_calculator.calculate_max_position_size(
    avg_volume_5min=current_bar["volume"]
)
actual_contracts = min(requested_contracts, max_contracts)

if actual_contracts < requested_contracts:
    logger.warning(f"Position reduced {requested_contracts}→{actual_contracts} due to volume")
```

**Verify:**
```bash
# Should see position reductions in low volume periods
python -m src backtest --log-level DEBUG | grep "Position reduced"
```

---

### 12A-5: Adverse Selection Bias

**Implementation:** Included in 12A-3 above (`apply_adverse_selection`)

---

### 12A-6: Integrate Bet Sizing (Task 4G)

**File:** `/Users/sneh/research/src/inference/backtesting/backtest.py`

**Modify line 397 in _calculate_position_size:**
```python
def _calculate_position_size(self, price: float, signal_confidence: float | None = None) -> int:
    """Calculate position size with optional bet sizing."""

    if signal_confidence is not None and self.config.use_bet_sizing:
        # Variable sizing by meta-labeling confidence
        return self.position_sizer.calculate_position_size(
            account_equity=self._equity,
            probability=signal_confidence,
            current_price=price,
        )
    else:
        # Standard fixed fractional / Kelly
        return self.position_sizer.calculate_position_size(
            account_equity=self._equity,
            current_price=price,
        )
```

**Add to BacktestConfig (src/config/inference.py line 50):**
```python
use_bet_sizing: bool = False  # Enable meta-labeling confidence sizing
```

**Verify:**
```bash
python -m src backtest --use-bet-sizing --meta-model path/to/meta.pkl
# Position sizes should vary by confidence
```

---

### 12A-7: Ensemble Diversity Penalty

**File (NEW):** `/Users/sneh/research/src/optimization/ensemble_objective.py`

**Create:**
```python
"""
Ensemble-aware Optuna objective with diversity penalty.

Optimizes: α × accuracy + β × diversity
where α=0.7, β=0.3
"""

from src.models.ensemble.diversity import compute_ensemble_diversity

def create_ensemble_objective(
    base_model_pool: list[str],
    oof_predictions: dict[str, np.ndarray],
    y_true: np.ndarray,
    alpha: float = 0.7,
    beta: float = 0.3,
):
    """
    Create Optuna objective that balances accuracy and diversity.

    Args:
        base_model_pool: Available models (e.g., ["xgboost", "lightgbm", "lstm"])
        oof_predictions: OOF predictions per model
        y_true: True labels
        alpha: Weight for accuracy (default 0.7)
        beta: Weight for diversity (default 0.3)
    """
    from itertools import combinations

    def objective(trial):
        # Select 3-5 models from pool
        n_models = trial.suggest_int("n_models", 3, min(5, len(base_model_pool)))
        selected_indices = trial.suggest_categorical(
            "selected_models",
            list(combinations(range(len(base_model_pool)), n_models))
        )
        selected_models = [base_model_pool[i] for i in selected_indices]

        # Get OOF predictions for selected models
        selected_oof = [oof_predictions[m] for m in selected_models]

        # Compute diversity
        diversity = compute_ensemble_diversity(
            selected_oof,
            method="q_statistic",  # or "disagreement", "entropy"
        )

        # Train meta-learner and evaluate
        from src.models.ensemble import StackingEnsemble
        ensemble = StackingEnsemble(base_models=selected_models)
        # ... train on OOF, evaluate
        accuracy = evaluate_meta(ensemble, selected_oof, y_true)

        # Combined objective
        return alpha * accuracy + beta * diversity

    return objective
```

**Usage:**
```bash
python -m src optimize-ensemble --models xgboost,lightgbm,catboost,lstm,gru,tcn
# Should select diverse subset
```

---

### 12A-8: Ensemble Feature Orthogonality

**File:** `/Users/sneh/research/src/optimization/five_dimension_objective.py`

**Add after line 450:**
```python
def constrain_feature_overlap(
    trial: optuna.Trial,
    model_id: str,
    previously_selected_features: set[str],
    all_features: list[str],
    max_overlap: float = 0.3,
) -> list[str]:
    """
    Enforce <30% feature overlap between ensemble models.

    Args:
        trial: Optuna trial
        model_id: Current model being optimized
        previously_selected_features: Features selected by other models
        all_features: Available feature pool
        max_overlap: Maximum fraction of overlap (default 0.3)

    Returns:
        Selected features with constrained overlap
    """
    selected = []
    overlap_count = 0

    for feature in all_features:
        if feature in previously_selected_features:
            # Lower probability for already-selected features
            use_prob = 0.3
        else:
            # Higher probability for fresh features
            use_prob = 0.7

        if trial.suggest_float(f"{model_id}_use_{feature}", 0, 1) < use_prob:
            selected.append(feature)
            if feature in previously_selected_features:
                overlap_count += 1

    # Check overlap constraint
    if previously_selected_features:
        overlap_pct = overlap_count / len(previously_selected_features)
        if overlap_pct > max_overlap:
            # Prune trial
            raise optuna.TrialPruned(f"Feature overlap {overlap_pct:.1%} > {max_overlap:.1%}")

    return selected
```

**Verify:**
```bash
python -m src test-feature-orthogonality
# Should show <30% overlap between model feature sets
```

---

## Phase 12B: Live Trading Safeguards (7 tasks)

### 12B-1: Drawdown Circuit Breaker (CRITICAL)

**File:** `/Users/sneh/research/src/inference/backtesting/backtest.py`

**Add to BacktestConfig (line 50):**
```python
max_drawdown_threshold: float = 0.10  # Halt trading at -10% drawdown
daily_loss_threshold: float = 0.02    # Halt at -2% daily loss
```

**Add in run() method after line 585:**
```python
# Check circuit breakers
current_drawdown = self.get_drawdowns()[-1] if len(self._equity_history) > 0 else 0

if abs(current_drawdown) > self.config.max_drawdown_threshold:
    logger.critical(
        f"🚨 CIRCUIT BREAKER TRIGGERED: "
        f"Drawdown {current_drawdown:.2%} exceeds threshold {self.config.max_drawdown_threshold:.2%}"
    )
    self._halt_trading = True
    self._liquidate_all_positions()
    break  # Stop trading loop

# Check daily loss
if bar_idx > 0 and self._is_new_day(current_bar, prev_bar):
    daily_return = (self._equity - self._day_start_equity) / self._day_start_equity
    if daily_return < -self.config.daily_loss_threshold:
        logger.critical(f"🚨 DAILY LOSS LIMIT: {daily_return:.2%}")
        self._halt_trading = True
        break
    self._day_start_equity = self._equity
```

**Add helper methods:**
```python
def _liquidate_all_positions(self):
    """Emergency liquidation of all open positions."""
    if self._current_position:
        self._close_position(
            reason="CIRCUIT_BREAKER",
            timestamp=self._current_position.entry_time,
            price=self._last_price,
        )

def _is_new_day(self, current_bar, prev_bar) -> bool:
    """Check if we crossed midnight."""
    return current_bar.name.date() != prev_bar.name.date()
```

**Verify:**
```bash
# Test with intentional bad strategy
python -m src backtest --max-drawdown-threshold 0.05 --strategy always_wrong
# Should trigger circuit breaker
```

---

### 12B-2: R-Multiple Tracking (CRITICAL)

**File:** `/Users/sneh/research/src/inference/backtesting/equity_curve.py`

**Modify Trade dataclass (line 25):**
```python
@dataclass
class Trade:
    # Existing fields...
    entry_price: float
    exit_price: float
    direction: int
    contracts: int
    net_pnl: float

    # NEW: R-multiple fields
    initial_risk_1r: float = 0.0     # Dollar risk at entry (stop distance × contracts)
    r_multiple: float = 0.0          # P&L / initial_risk_1r
    stop_loss_price: float | None = None

    def calculate_r_multiple(self, point_value: float = 5.0):
        """Calculate R-multiple from stop loss."""
        if self.stop_loss_price is None or self.initial_risk_1r == 0:
            self.r_multiple = 0.0
            return

        # 1R = initial risk
        risk_per_contract = abs(self.entry_price - self.stop_loss_price) * point_value
        self.initial_risk_1r = risk_per_contract * self.contracts

        # R-multiple = P&L / initial risk
        if self.initial_risk_1r > 0:
            self.r_multiple = self.net_pnl / self.initial_risk_1r
```

**Update backtest.py to set stop_loss_price:**
```python
# Line 410 in _open_position
atr = self._get_current_atr(bar_idx)  # Need ATR column in data
stop_distance_atr = 2.0  # 2 ATR stop
stop_loss = price - direction * stop_distance_atr * atr

self._current_position = Position(
    direction=direction,
    contracts=contracts,
    entry_price=price,
    stop_loss=stop_loss,  # NOW SET
    # ...
)
```

**Calculate R when closing trade:**
```python
# Line 505 in _close_position
trade = Trade(
    entry_price=position.entry_price,
    exit_price=price,
    stop_loss_price=position.stop_loss,  # Pass stop
    # ...
)
trade.calculate_r_multiple(point_value=self.config.costs.tick_value * 4)
```

**Verify:**
```bash
python -m src backtest --output results/
python -c "import pandas as pd; t = pd.read_csv('results/trades.csv'); print(f'Avg Win R: {t[t.r_multiple>0].r_multiple.mean():.2f}R')"
# Should print R-multiples
```

---

### 12B-3: Enforce Stop Losses

**Implementation:** Included in 12B-2 above

---

### 12B-4: Daily Loss Limits

**Implementation:** Included in 12B-1 above

---

### 12B-5: R-Based Expectancy

**File (NEW):** `/Users/sneh/research/src/inference/backtesting/r_analysis.py`

**Create:**
```python
"""
R-multiple analysis for trading strategies.

Provides:
- Expectancy in R: E = (Win% × Avg Win R) - (Loss% × Avg Loss R)
- R-multiple distribution
- Kelly fraction from R distribution
- Risk of ruin calculation
"""

import numpy as np
import pandas as pd

def calculate_r_expectancy(trades: pd.DataFrame) -> dict:
    """Calculate expectancy in R-multiples."""
    if "r_multiple" not in trades.columns:
        raise ValueError("Trades must have r_multiple column")

    wins = trades[trades.r_multiple > 0]
    losses = trades[trades.r_multiple <= 0]

    win_rate = len(wins) / len(trades) if len(trades) > 0 else 0
    loss_rate = 1 - win_rate

    avg_win_r = wins.r_multiple.mean() if len(wins) > 0 else 0
    avg_loss_r = abs(losses.r_multiple.mean()) if len(losses) > 0 else 0

    # E = (Win% × Avg Win R) - (Loss% × Avg Loss R)
    expectancy_r = (win_rate * avg_win_r) - (loss_rate * avg_loss_r)

    return {
        "win_rate": win_rate,
        "avg_win_r": avg_win_r,
        "avg_loss_r": avg_loss_r,
        "expectancy_r": expectancy_r,
        "profit_factor_r": (win_rate * avg_win_r) / (loss_rate * avg_loss_r + 1e-8),
    }

def calculate_kelly_from_r(trades: pd.DataFrame) -> float:
    """Calculate Kelly fraction from R distribution."""
    r_stats = calculate_r_expectancy(trades)

    # Kelly = (Win% × Avg Win R - Loss% × Avg Loss R) / Avg Win R
    # Simplified: f* = p - q/b where b = avg_win_r / avg_loss_r
    p = r_stats["win_rate"]
    b = r_stats["avg_win_r"] / (r_stats["avg_loss_r"] + 1e-8)

    kelly = p - (1 - p) / b
    return max(0, kelly)  # No negative Kelly
```

**Verify:**
```bash
python -c "
from src.inference.backtesting.r_analysis import calculate_r_expectancy
import pandas as pd
trades = pd.read_csv('results/trades.csv')
stats = calculate_r_expectancy(trades)
print(f'Expectancy: {stats[\"expectancy_r\"]:.3f}R per trade')
"
```

---

### 12B-6: Monte Carlo Stress Testing

**File (NEW):** `/Users/sneh/research/src/inference/backtesting/monte_carlo.py`

**Create:**
```python
"""
Monte Carlo simulation and stress testing for trading strategies.
"""

import numpy as np
from typing import Callable

def monte_carlo_simulation(
    trades: list[dict],  # List of trade P&Ls
    initial_equity: float,
    n_simulations: int = 10000,
) -> dict:
    """
    Bootstrap resample trades to estimate outcome distribution.

    Returns:
        - worst_case_dd_95: 95th percentile worst drawdown
        - median_return: Median final return
        - risk_of_ruin: P(drawdown > 50%)
        - sharpe_distribution: Distribution of Sharpe ratios
    """
    worst_drawdowns = []
    final_returns = []
    ruin_count = 0

    for _ in range(n_simulations):
        # Resample trades with replacement
        sampled_trades = np.random.choice(trades, size=len(trades), replace=True)

        # Simulate equity curve
        equity = initial_equity
        equity_curve = [equity]

        for trade in sampled_trades:
            equity += trade["net_pnl"]
            equity_curve.append(equity)

        # Calculate max drawdown
        equity_arr = np.array(equity_curve)
        running_max = np.maximum.accumulate(equity_arr)
        drawdowns = (equity_arr - running_max) / running_max
        max_dd = drawdowns.min()

        worst_drawdowns.append(max_dd)
        final_returns.append((equity - initial_equity) / initial_equity)

        if max_dd < -0.50:  # 50% drawdown = ruin
            ruin_count += 1

    return {
        "worst_case_dd_95": np.percentile(worst_drawdowns, 95),
        "median_return": np.median(final_returns),
        "risk_of_ruin": ruin_count / n_simulations,
        "sharpe_distribution": final_returns,  # Can compute Sharpe from this
    }

def stress_test_scenarios(
    strategy: Callable,
    prices: pd.DataFrame,
    scenarios: dict[str, dict],
) -> dict:
    """
    Test strategy under extreme scenarios.

    Scenarios:
        - flash_crash: 5x volatility spike
        - trending: Filter to trending regimes only
        - choppy: Filter to mean-reverting regimes
        - low_volume: 3x slippage
    """
    results = {}

    for scenario_name, params in scenarios.items():
        # Apply scenario transformations
        if scenario_name == "flash_crash":
            modified_prices = prices.copy()
            modified_prices["high"] *= 1.05
            modified_prices["low"] *= 0.95
        elif scenario_name == "low_volume":
            # Increase slippage
            pass

        # Run strategy
        result = strategy(modified_prices, **params)
        results[scenario_name] = result

    return results
```

**Verify:**
```bash
python -m src monte-carlo --trades results/trades.csv --simulations 10000
# Should output: "95th percentile worst DD: -12.5%"
```

---

### 12B-7: Portfolio Risk Aggregation

**File (NEW):** `/Users/sneh/research/src/inference/backtesting/portfolio_risk.py`

**Create:**
```python
"""
Portfolio-level risk management.

Enforces:
- Max leverage (total notional / equity)
- Correlation-based position sizing
- Sector concentration limits
"""

from dataclasses import dataclass

@dataclass
class PortfolioRiskLimits:
    max_leverage: float = 1.0          # No leverage
    max_correlation: float = 0.7       # Reduce correlated positions
    max_sector_pct: float = 0.30       # Max 30% in one sector

class PortfolioRiskManager:
    def __init__(self, limits: PortfolioRiskLimits):
        self.limits = limits

    def check_new_position(
        self,
        new_position_notional: float,
        existing_positions: list[dict],
        account_equity: float,
    ) -> tuple[bool, float]:
        """
        Check if new position violates portfolio limits.

        Returns:
            (can_trade, size_adjustment_factor)
        """
        # Check total leverage
        total_notional = sum(p["notional"] for p in existing_positions)
        total_notional += new_position_notional

        if total_notional > self.limits.max_leverage * account_equity:
            # Reduce position to stay within leverage
            max_new_notional = (self.limits.max_leverage * account_equity) - sum(p["notional"] for p in existing_positions)
            if max_new_notional <= 0:
                return False, 0.0
            adjustment = max_new_notional / new_position_notional
            return True, adjustment

        return True, 1.0
```

**Verify:**
```bash
python -c "
from src.inference.backtesting.portfolio_risk import PortfolioRiskManager, PortfolioRiskLimits
rm = PortfolioRiskManager(PortfolioRiskLimits(max_leverage=1.0))
can_trade, adj = rm.check_new_position(100000, [{'notional': 50000}], 100000)
print(f'Can trade: {can_trade}, Adjustment: {adj}')
"
```

---

## Phase 12C: Deployment Infrastructure (6 tasks)

### 12C-1: Enable MLflow by Default (5 min fix!)

**File:** `/Users/sneh/research/src/config/training.py`

**Line 398:**
```python
# BEFORE:
tracking_enabled: bool = True
tracking_backend: str = "local"  # Uses JSON files

# AFTER:
tracking_enabled: bool = True
tracking_backend: str = "mlflow"  # Auto-enable MLflow
```

**Verify:**
```bash
# Start MLflow server
mlflow server --backend-store-uri sqlite:///mlruns.db --port 5000 &

# Train model (should auto-log)
python -m src train --symbol MES --models xgboost --horizons 20

# Check MLflow UI
open http://localhost:5000
# Should see logged run
```

---

### 12C-2: Integrate Drift Monitoring into Inference

**File:** `/Users/sneh/research/src/inference/pipeline.py`

**Add after line 150 in predict():**
```python
def predict(self, X, calibrate=True):
    # ... existing preprocessing ...

    # Drift detection
    if self.drift_monitor and len(self._prediction_history) > 100:
        drift_results = self.drift_monitor.check_drift(X_array)

        for result in drift_results:
            if result.drift_detected and result.severity in [DriftSeverity.HIGH, DriftSeverity.CRITICAL]:
                logger.warning(
                    f"⚠️  DRIFT DETECTED: {result.feature_name} "
                    f"(severity: {result.severity.value}, "
                    f"metric: {result.metric_value:.4f})"
                )

                # Optional: Include in output metadata
                if not hasattr(output, "metadata"):
                    output.metadata = {}
                output.metadata["drift_warnings"] = [
                    {"feature": r.feature_name, "severity": r.severity.value}
                    for r in drift_results if r.drift_detected
                ]

    # Make prediction
    # ... existing code ...
```

**Add to __init__:**
```python
def __init__(self, model_bundle, enable_monitoring=True):
    # ... existing init ...

    if enable_monitoring:
        from src.validation.monitoring import FeatureDriftMonitor
        self.drift_monitor = FeatureDriftMonitor(
            method="psi",
            threshold=0.2,
        )
        # Fit on training data from bundle
        if hasattr(model_bundle, "_training_reference_data"):
            self.drift_monitor.fit(model_bundle._training_reference_data)
    else:
        self.drift_monitor = None
```

**Verify:**
```bash
python -c "
from src.inference.pipeline import InferencePipeline
from src.inference.bundle import ModelBundle
bundle = ModelBundle.load('experiments/run_001/models/xgboost_h20/')
pipeline = InferencePipeline(bundle, enable_monitoring=True)
# Predict with drifted data
import numpy as np
X_drifted = np.random.randn(100, 180) * 10  # 10x scaled
result = pipeline.predict(X_drifted)
# Should see drift warnings in logs
"
```

---

### 12C-3: Add ProductionMonitor

**File (NEW):** `/Users/sneh/research/src/inference/monitor.py`

**Create:**
```python
"""
Production model monitoring.

Tracks:
- Feature drift (PSI, KS tests)
- Model performance degradation
- Model freshness (time since training)
- Prediction distribution shifts
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np

from src.validation.monitoring import FeatureDriftMonitor, DriftSeverity

@dataclass
class ModelHealthMetrics:
    accuracy: float
    drift_features_count: int
    model_age_days: int
    avg_latency_ms: float
    error_rate: float
    last_checked: datetime

class ProductionMonitor:
    """Unified production monitoring for deployed models."""

    def __init__(
        self,
        model_bundle,
        baseline_accuracy: float = 0.60,
        max_model_age_days: int = 30,
        alert_handler = None,
    ):
        self.bundle = model_bundle
        self.baseline_accuracy = baseline_accuracy
        self.max_model_age_days = max_model_age_days
        self.alert_handler = alert_handler

        # Drift monitor
        self.drift_monitor = FeatureDriftMonitor(method="psi", threshold=0.2)

        # Performance tracking
        self.prediction_history = []

    def check_model_freshness(self) -> tuple[bool, str]:
        """Check if model is too old."""
        created_at = datetime.fromisoformat(self.bundle._metadata.created_at)
        age_days = (datetime.now() - created_at).days

        if age_days > self.max_model_age_days:
            msg = f"Model age {age_days} days exceeds max {self.max_model_age_days}"
            if self.alert_handler:
                self.alert_handler.send_alert("model_freshness", "HIGH", msg)
            return False, msg

        return True, f"Model age: {age_days} days (OK)"

    def check_performance_degradation(self, y_true, y_pred) -> tuple[bool, dict]:
        """Check if accuracy has degraded."""
        from sklearn.metrics import accuracy_score

        current_acc = accuracy_score(y_true, y_pred)

        if current_acc < self.baseline_accuracy * 0.80:  # 20% drop
            msg = f"Accuracy degraded: {self.baseline_accuracy:.3f} → {current_acc:.3f}"
            if self.alert_handler:
                self.alert_handler.send_alert("accuracy_degradation", "CRITICAL", msg)
            return False, {"current": current_acc, "baseline": self.baseline_accuracy}

        return True, {"current": current_acc, "baseline": self.baseline_accuracy}

    def run_all_checks(self, X_current, y_true=None, y_pred=None) -> ModelHealthMetrics:
        """Run all monitoring checks."""
        freshness_ok, freshness_msg = self.check_model_freshness()

        drift_results = self.drift_monitor.check_drift(X_current)
        drift_count = sum(1 for r in drift_results if r.drift_detected)

        perf_ok, perf_metrics = True, {}
        if y_true is not None and y_pred is not None:
            perf_ok, perf_metrics = self.check_performance_degradation(y_true, y_pred)

        created_at = datetime.fromisoformat(self.bundle._metadata.created_at)
        age_days = (datetime.now() - created_at).days

        return ModelHealthMetrics(
            accuracy=perf_metrics.get("current", 0.0),
            drift_features_count=drift_count,
            model_age_days=age_days,
            avg_latency_ms=0.0,  # TODO: track from inference pipeline
            error_rate=0.0,
            last_checked=datetime.now(),
        )
```

**Verify:**
```bash
python -c "
from src.inference.monitor import ProductionMonitor
from src.inference.bundle import ModelBundle
bundle = ModelBundle.load('experiments/run_001/models/xgboost_h20/')
monitor = ProductionMonitor(bundle, baseline_accuracy=0.65)
import numpy as np
X = np.random.randn(100, 180)
health = monitor.run_all_checks(X)
print(f'Model age: {health.model_age_days} days, Drift features: {health.drift_features_count}')
"
```

---

### 12C-4: Slack/PagerDuty Alerts

**File (NEW):** `/Users/sneh/research/src/validation/monitoring/connectors/slack.py`

**Create:**
```python
"""
Slack alert connector for drift and performance alerts.
"""

from dataclasses import dataclass

@dataclass
class SlackConfig:
    token: str
    channel: str = "#model-alerts"
    username: str = "ML Factory Monitor"
    icon_emoji: str = ":robot_face:"

class SlackAlertConnector:
    """Send alerts to Slack channel."""

    def __init__(self, config: SlackConfig):
        self.config = config

        try:
            from slack_sdk import WebClient
            self.client = WebClient(token=config.token)
        except ImportError:
            raise ImportError("slack-sdk not installed. Install with: pip install slack-sdk")

    def send_drift_alert(self, feature_name: str, severity: str, metric_value: float, threshold: float):
        """Send drift detection alert."""
        severity_emoji = {
            "LOW": ":large_yellow_circle:",
            "MEDIUM": ":large_orange_circle:",
            "HIGH": ":red_circle:",
            "CRITICAL": ":rotating_light:",
        }

        message = {
            "text": f"{severity_emoji.get(severity, ':warning:')} *DRIFT ALERT*",
            "blocks": [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Feature drift detected: {feature_name}*"
                    }
                },
                {
                    "type": "section",
                    "fields": [
                        {"type": "mrkdwn", "text": f"*Severity:*\n{severity}"},
                        {"type": "mrkdwn", "text": f"*PSI Metric:*\n{metric_value:.4f}"},
                        {"type": "mrkdwn", "text": f"*Threshold:*\n{threshold:.4f}"},
                    ]
                },
            ]
        }

        try:
            self.client.chat_postMessage(
                channel=self.config.channel,
                username=self.config.username,
                icon_emoji=self.config.icon_emoji,
                **message
            )
        except Exception as e:
            print(f"Failed to send Slack alert: {e}")
```

**Add to pyproject.toml:**
```toml
[project.optional-dependencies]
monitoring = [
    "slack-sdk>=3.26.0",
    "prometheus-client>=0.19.0",
]
```

**Verify:**
```bash
export SLACK_BOT_TOKEN="xoxb-your-token"
python -c "
from src.validation.monitoring.connectors.slack import SlackAlertConnector, SlackConfig
import os
config = SlackConfig(token=os.getenv('SLACK_BOT_TOKEN'))
connector = SlackAlertConnector(config)
connector.send_drift_alert('rsi_14', 'HIGH', 0.35, 0.20)
"
# Should post to Slack
```

---

### 12C-5: Feature Distribution Validation

**File:** `/Users/sneh/research/src/inference/bundle.py`

**Add method after line 706:**
```python
def validate_distribution(
    self,
    X_current: pd.DataFrame,
    method: str = "ks",  # "ks", "psi", or "quantile"
    threshold: float = 0.05,
) -> tuple[bool, list[str]]:
    """
    Compare feature distributions to training data.

    Args:
        X_current: Current inference data
        method: Statistical test ("ks", "psi", "quantile")
        threshold: p-value threshold (for KS) or PSI threshold

    Returns:
        (is_valid, list_of_warnings)
    """
    if not hasattr(self, "_training_stats"):
        return True, ["No training stats available for comparison"]

    warnings = []

    for i, feature_name in enumerate(self.feature_columns):
        if feature_name not in X_current.columns:
            continue

        current_values = X_current[feature_name].dropna()
        train_stats = self._training_stats.get(feature_name, {})

        if method == "ks":
            from scipy.stats import ks_2samp
            # Compare to training distribution
            train_mean = train_stats.get("mean", 0)
            train_std = train_stats.get("std", 1)

            # Standardize current
            current_std = (current_values - train_mean) / (train_std + 1e-8)

            # KS test against standard normal (since training was standardized)
            stat, p_value = ks_2samp(current_std, np.random.randn(1000))

            if p_value < threshold:
                warnings.append(
                    f"{feature_name}: Distribution shift detected "
                    f"(KS p={p_value:.4f} < {threshold})"
                )

        elif method == "psi":
            # Population Stability Index
            train_quantiles = train_stats.get("quantiles", [])
            # ... PSI calculation
            pass

    return len(warnings) == 0, warnings
```

**Store training stats in bundle creation:**
```python
# When creating bundle, save training feature stats
self._training_stats = {}
for col in X_train.columns:
    self._training_stats[col] = {
        "mean": X_train[col].mean(),
        "std": X_train[col].std(),
        "min": X_train[col].min(),
        "max": X_train[col].max(),
        "quantiles": X_train[col].quantile([0.05, 0.25, 0.5, 0.75, 0.95]).tolist(),
    }
```

**Verify:**
```bash
python -c "
from src.inference.bundle import ModelBundle
import pandas as pd
import numpy as np
bundle = ModelBundle.load('experiments/run_001/models/xgboost_h20/')
# Simulate drift
X_drifted = pd.DataFrame(np.random.randn(100, 180) * 5, columns=bundle.feature_columns)
is_valid, warnings = bundle.validate_distribution(X_drifted, method='ks')
print(f'Valid: {is_valid}, Warnings: {len(warnings)}')
"
```

---

### 12C-6: Prometheus Metrics Export

**File:** `/Users/sneh/research/src/inference/server.py`

**Add after line 30:**
```python
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

# Prometheus metrics
request_count = Counter('inference_requests_total', 'Total inference requests', ['model_name', 'status'])
latency_histogram = Histogram('inference_latency_seconds', 'Inference latency', ['model_name'])
drift_gauge = Gauge('feature_drift_psi', 'PSI drift metric', ['feature_name'])
accuracy_gauge = Gauge('model_accuracy', 'Current model accuracy')
```

**Add endpoint after line 380:**
```python
@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
```

**Update predict endpoint to track metrics:**
```python
# Line 200 in predict endpoint
start_time = time.time()

try:
    result = self.pipeline.predict(X)
    latency_histogram.labels(model_name=self.pipeline.model_name).observe(time.time() - start_time)
    request_count.labels(model_name=self.pipeline.model_name, status="success").inc()
except Exception as e:
    request_count.labels(model_name=self.pipeline.model_name, status="error").inc()
    raise
```

**Add to pyproject.toml:**
```toml
prometheus-client>=0.19.0
```

**Verify:**
```bash
# Start server
uvicorn src.inference.server:app --port 8000 &

# Make prediction
curl -X POST http://localhost:8000/predict -d '{"features": [[...]]}'

# Check Prometheus metrics
curl http://localhost:8000/metrics
# Should see: inference_requests_total, inference_latency_seconds, etc.
```

---

## Phase 12D: Pipeline Performance (7 tasks)

### 12D-1: Enable Parallel Training (5 min fix!)

**File:** `/Users/sneh/research/src/models/training/unified_orchestrator.py`

**Line 80 - Change service initialization:**
```python
# BEFORE:
self._model_service = ModelTrainingService()  # Sequential

# AFTER:
from src.models.training.services.parallel_training import ParallelTrainingService
self._model_service = ParallelTrainingService(n_jobs=-1)  # Use all CPUs
```

**Verify:**
```bash
# Watch CPU usage
htop &

# Train multiple models
python -m src train --models xgboost,lightgbm,catboost --horizons 20,60

# Should see all CPU cores at 100%
```

---

### 12D-2: Integrate FeatureStore (HIGHEST PERFORMANCE IMPACT)

**File:** `/Users/sneh/research/src/data/pipeline/stages/features/run.py`

**Add imports at top:**
```python
from src.data.store import FeatureStore
from src.core.lineage import compute_file_checksum, compute_config_hash
```

**Modify engineer_features_stage() starting at line 123:**
```python
def engineer_features_stage(config: DataConfig) -> None:
    """Engineer features with caching."""

    # Initialize FeatureStore
    cache_dir = config.output_dir / "feature_cache"
    store = FeatureStore(cache_dir=cache_dir)

    logger.info("🔧 Feature Engineering Stage (with caching)")

    for symbol in config.symbols:
        for tf in output_timeframes:
            # Compute cache key
            input_file = config.clean_data_dir / f"{symbol}_{tf}_clean.parquet"
            input_checksum = compute_file_checksum(input_file)
            config_hash = compute_config_hash({
                "timeframe": tf,
                "mtf_timeframes": config.mtf_timeframes,
                "feature_toggles": effective_toggles,
                # ... all relevant config
            })

            feature_set = f"pipeline_{tf}_{config_hash[:8]}"

            # Check cache
            if store.has_features(symbol=symbol, feature_set=feature_set):
                logger.info(f"✓ CACHE HIT: {symbol}@{tf} (loading cached features)")
                df_features = store.get_features(symbol=symbol, feature_set=feature_set)
            else:
                logger.info(f"  CACHE MISS: {symbol}@{tf} (computing features)")

                # Load data
                df = pd.read_parquet(input_file)

                # Engineer features
                engineer = FeatureEngineer(...)
                df_features, feature_info = engineer.engineer_features(df, symbol)

                # Store in cache
                store.put_features(
                    df_features,
                    symbol=symbol,
                    feature_set=feature_set,
                    lineage={
                        "raw_path": str(input_file),
                        "input_checksum": input_checksum,
                        "config": effective_toggles,
                        "transformations": feature_info.get("transformations", []),
                    },
                    version="1.0.0",
                )

            # Save to output
            output_file = config.output_dir / f"{symbol}_{tf}_features.parquet"
            df_features.to_parquet(output_file, index=False)

    logger.info("✅ Feature Engineering Complete")
```

**Add config field:**
```python
# src/data/pipeline/data_config.py
feature_cache_dir: Path | None = None  # Default: {output_dir}/feature_cache
enable_feature_caching: bool = True
```

**Verify:**
```bash
# First run (cold cache)
time python -m src pipeline --stages features --symbol MES
# Note time: ~5 min

# Second run (warm cache)
time python -m src pipeline --stages features --symbol MES
# Should be <10 seconds (50x faster!)

# Check cache
ls -lh data/output/feature_cache/
```

---

### 12D-3: Parallelize Optuna Trials

**File:** `/Users/sneh/research/src/optimization/five_dimension_objective.py`

**Line 879 - Add n_jobs parameter:**
```python
# BEFORE:
study.optimize(objective, n_trials=config.n_trials, direction="maximize")

# AFTER:
import os
n_jobs = config.n_jobs if hasattr(config, 'n_jobs') else os.cpu_count()
logger.info(f"Running Optuna with {n_jobs} parallel trials")

study.optimize(
    objective,
    n_trials=config.n_trials,
    direction="maximize",
    n_jobs=n_jobs,  # Parallelize!
)
```

**Add to config:**
```python
# src/optimization/config.py or src/core/config.py
n_jobs: int = -1  # Use all CPUs (-1), or specific number
```

**Verify:**
```bash
# Watch CPU/GPU usage
htop &
nvidia-smi -l 1 &

# Run optimization
python -m src optimize --trials 100 --n-jobs -1

# Should see multiple trials running in parallel
```

---

### 12D-4: Enable GPU for Boosting Models

**Files:**
- `/Users/sneh/research/src/models/boosting/xgboost_model.py`
- `/Users/sneh/research/src/models/boosting/lightgbm_model.py`
- `/Users/sneh/research/src/models/boosting/catboost_model.py`

**XGBoost - Modify get_default_params():**
```python
# Line ~150
def get_default_params(self) -> dict:
    return {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 6,
        # ADD GPU PARAMS:
        "tree_method": "gpu_hist",      # Use GPU
        "predictor": "gpu_predictor",   # GPU prediction
        "gpu_id": 0,                     # First GPU
    }
```

**LightGBM - Modify get_default_params():**
```python
def get_default_params(self) -> dict:
    return {
        "n_estimators": 100,
        "learning_rate": 0.1,
        # ADD GPU PARAMS:
        "device": "gpu",
        "gpu_platform_id": 0,
        "gpu_device_id": 0,
    }
```

**CatBoost - Modify get_default_params():**
```python
def get_default_params(self) -> dict:
    return {
        "iterations": 100,
        "learning_rate": 0.1,
        # ADD GPU PARAMS:
        "task_type": "GPU",
        "devices": "0",  # GPU ID
    }
```

**Verify:**
```bash
# Should show GPU usage
nvidia-smi -l 1 &
python -m src train --models xgboost,lightgbm,catboost

# GPU memory should increase during training
```

---

### 12D-5: Numba Parallel Labeling

**File:** `/Users/sneh/research/src/data/labeling/triple_barrier.py`

**Line 232 - Modify decorator:**
```python
# BEFORE:
@nb.jit(nopython=True, cache=True)
def _compute_barriers_vectorized(...):

# AFTER:
@nb.jit(nopython=True, cache=True, parallel=True)
def _compute_barriers_vectorized(...):
```

**Line 160 - Use prange:**
```python
# BEFORE:
for i in range(n - 1):

# AFTER:
for i in nb.prange(n - 1):  # Parallel range
```

**Verify:**
```bash
# Should use multiple cores
htop &
python -m src pipeline --stages final_labels
# CPU usage should be distributed across cores
```

---

### 12D-6: Cache MTF Upsampled Data

**File:** `/Users/sneh/research/src/data/pipeline/stages/mtf/__init__.py`

**Add caching:**
```python
from functools import lru_cache
import hashlib

@lru_cache(maxsize=100)
def get_cached_upsampled_data(
    source_file: str,
    source_mtime: float,
    target_timeframe: str,
) -> pd.DataFrame:
    """Cache upsampled data keyed by source file + mtime."""
    df = pd.read_parquet(source_file)
    upsampled = upsample_to_timeframe(df, target_timeframe)
    return upsampled

def generate_mtf_stage(config: DataConfig):
    for symbol in config.symbols:
        for source_tf in config.mtf_timeframes:
            source_file = config.clean_data_dir / f"{symbol}_{source_tf}_clean.parquet"
            source_mtime = source_file.stat().st_mtime

            for target_tf in higher_timeframes:
                # Use cache
                df_upsampled = get_cached_upsampled_data(
                    str(source_file),
                    source_mtime,
                    target_tf,
                )
```

**Verify:**
```bash
# First run
time python -m src pipeline --stages mtf
# Second run (should be faster)
time python -m src pipeline --stages mtf
```

---

### 12D-7: Stage Timeout Protection

**File:** `/Users/sneh/research/src/data/pipeline/utils.py` (NEW)

**Create:**
```python
"""
Pipeline utilities including timeout protection.
"""

import signal
from contextlib import contextmanager
from typing import Callable

class TimeoutError(Exception):
    pass

@contextmanager
def timeout(seconds: int, stage_name: str = ""):
    """Context manager for stage timeout."""
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Stage '{stage_name}' exceeded timeout of {seconds}s")

    # Set alarm
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)

    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

def run_stage_with_timeout(
    stage_fn: Callable,
    timeout_seconds: int,
    stage_name: str,
    *args,
    **kwargs,
):
    """Run pipeline stage with timeout protection."""
    logger.info(f"Running {stage_name} (timeout: {timeout_seconds}s)")

    try:
        with timeout(timeout_seconds, stage_name):
            result = stage_fn(*args, **kwargs)
        logger.info(f"✓ {stage_name} completed")
        return result
    except TimeoutError as e:
        logger.error(f"✗ {stage_name} TIMEOUT: {e}")
        raise
```

**Use in pipeline runner:**
```python
# src/data/pipeline/runner.py
from src.data.pipeline.utils import run_stage_with_timeout

# Wrap each stage
run_stage_with_timeout(
    ingest_stage,
    timeout_seconds=1800,  # 30 min
    stage_name="ingest",
    config=config,
)
```

**Verify:**
```bash
# Should timeout after 10s
python -c "
from src.data.pipeline.utils import timeout
import time
with timeout(10, 'test'):
    time.sleep(20)  # Will raise TimeoutError
"
```

---

## Phase 12E: Testing Infrastructure (5 tasks)

### 12E-1: Test Cross-Validation (CRITICAL)

**File (NEW):** `/Users/sneh/research/tests/test_validation.py`

**Create:**
```python
"""
Unit tests for cross-validation and leakage prevention.
"""

import pytest
import numpy as np
import pandas as pd
from src.validation.cv import PurgedKFold

def create_test_data(n=1000, horizon=20):
    """Create synthetic labeled data for testing."""
    dates = pd.date_range("2020-01-01", periods=n, freq="5min")
    df = pd.DataFrame({
        "close": np.random.randn(n).cumsum() + 100,
        "label": np.random.choice([-1, 0, 1], n),
    }, index=dates)
    return df

def test_purged_kfold_no_overlap():
    """Verify train/test labels don't overlap within horizon."""
    df = create_test_data(n=1000, horizon=20)
    cv = PurgedKFold(n_splits=5, purge_bars=60, embargo_bars=100)

    for train_idx, test_idx in cv.split(df):
        # Get timestamps
        train_times = df.index[train_idx]
        test_times = df.index[test_idx]

        # Check: No train sample within horizon of test start
        test_start = test_times.min()
        too_close = train_times[train_times >= test_start - pd.Timedelta(minutes=20*5)]

        assert len(too_close) == 0, f"Found {len(too_close)} train samples within horizon of test"

def test_purged_kfold_embargo_applied():
    """Verify embargo bars are applied after test set."""
    df = create_test_data(n=1000, horizon=20)
    cv = PurgedKFold(n_splits=5, purge_bars=60, embargo_bars=100)

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(df)):
        test_end = df.index[test_idx].max()

        # Check: No train samples within embargo_bars of test end
        embargo_start = test_end
        embargo_end = embargo_start + pd.Timedelta(minutes=100*5)

        train_in_embargo = df.index[train_idx][(df.index[train_idx] > embargo_start) & (df.index[train_idx] < embargo_end)]

        assert len(train_in_embargo) == 0, f"Fold {fold_idx}: Found {len(train_in_embargo)} train samples in embargo period"

def test_purged_kfold_coverage():
    """Verify each sample appears in test set exactly once."""
    df = create_test_data(n=1000)
    cv = PurgedKFold(n_splits=5, purge_bars=60, embargo_bars=100)

    test_counts = np.zeros(len(df))

    for train_idx, test_idx in cv.split(df):
        test_counts[test_idx] += 1

    # Each sample should appear in test set at most once
    assert np.max(test_counts) <= 1, "Some samples appear in multiple test folds"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

**Verify:**
```bash
pytest tests/test_validation.py -v
```

---

### 12E-2: Test Adapters

**File (NEW):** `/Users/sneh/research/tests/test_adapters.py`

**Create:**
```python
"""
Unit tests for data adapters.
"""

import pytest
import numpy as np
import pandas as pd
from src.data.adapters import TabularAdapter, SequenceAdapter, MultiStreamAdapter
from src.core.contracts import ModelContract, DataContract

def create_test_features(n_samples=100, n_features=180):
    """Create synthetic feature DataFrame."""
    return pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feature_{i}" for i in range(n_features)]
    )

def test_tabular_adapter_shape():
    """Verify TabularAdapter output shape matches contract."""
    df = create_test_features(n_samples=100, n_features=180)
    df["label"] = np.random.choice([-1, 0, 1], 100)

    contract = ModelContract(
        model_name="xgboost",
        data_rank=2,  # 2D tabular
        input_shape=(None, 180),
        output_shape=(None, 3),
    )

    adapter = TabularAdapter()
    result = adapter.transform(df, model_contract=contract)

    # Check shapes
    assert result.X.shape == (100, 180), f"Expected (100, 180), got {result.X.shape}"
    assert result.y.shape == (100,), f"Expected (100,), got {result.y.shape}"

    # Check data contract
    assert result.data_contract.n_features == 180
    assert result.data_contract.n_samples == 100

def test_sequence_adapter_shape():
    """Verify SequenceAdapter creates correct 3D tensors."""
    df = create_test_features(n_samples=200, n_features=50)
    df["label"] = np.random.choice([-1, 0, 1], 200)

    contract = ModelContract(
        model_name="lstm",
        data_rank=3,  # 3D sequences
        input_shape=(None, 20, 50),  # (batch, sequence_length, features)
        output_shape=(None, 3),
    )

    adapter = SequenceAdapter(sequence_length=20)
    result = adapter.transform(df, model_contract=contract)

    # Should have (n_samples - sequence_length + 1) sequences
    expected_samples = 200 - 20 + 1
    assert result.X.shape == (expected_samples, 20, 50), f"Expected ({expected_samples}, 20, 50), got {result.X.shape}"
    assert result.y.shape == (expected_samples,)

def test_multi_stream_adapter_4d():
    """Verify MultiStreamAdapter creates 4D tensors for transformers."""
    # Create multi-timeframe data
    dfs = {
        "5m": create_test_features(200, 50),
        "15m": create_test_features(200, 50),
        "60m": create_test_features(200, 50),
    }
    for df in dfs.values():
        df["label"] = np.random.choice([-1, 0, 1], 200)

    contract = ModelContract(
        model_name="patchtst",
        data_rank=4,  # 4D multi-timeframe
        input_shape=(None, 3, 50, 50),  # (batch, timeframes, features, channels)
        output_shape=(None, 3),
    )

    adapter = MultiStreamAdapter()
    result = adapter.transform(dfs["5m"], model_contract=contract, additional_dfs=dfs)

    # Check 4D shape
    assert len(result.X.shape) == 4, f"Expected 4D tensor, got {len(result.X.shape)}D"
    assert result.X.shape[1] == 3, f"Expected 3 timeframes, got {result.X.shape[1]}"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

**Verify:**
```bash
pytest tests/test_adapters.py -v
```

---

### 12E-3: Test Leakage Detection

**File (NEW):** `/Users/sneh/research/tests/test_leakage.py`

**Create:**
```python
"""
Unit tests for leakage detection.
"""

import pytest
import numpy as np
import pandas as pd
from src.validation.leakage_detection import detect_leakage

def create_leaked_data():
    """Create data with known leakage (feature = future label)."""
    n = 1000
    df = pd.DataFrame({
        "feature_1": np.random.randn(n),
        "feature_2": np.random.randn(n),
        "label": np.random.choice([0, 1], n),
    })

    # Introduce leakage: feature_leaked = label shifted forward
    df["feature_leaked"] = df["label"].shift(-1).fillna(0)

    return df

def test_detect_temporal_leakage():
    """Verify temporal leakage detection catches future data."""
    df = create_leaked_data()

    results = detect_leakage(
        df,
        label_col="label",
        feature_cols=["feature_1", "feature_2", "feature_leaked"],
        methods=["temporal"],
    )

    # Should detect feature_leaked as leaky
    assert "feature_leaked" in results["temporal"]["leaky_features"], \
        "Failed to detect temporal leakage in feature_leaked"

def test_detect_correlation_leakage():
    """Verify correlation-based leakage detection."""
    n = 1000
    df = pd.DataFrame({
        "feature_normal": np.random.randn(n),
        "label": np.random.choice([0, 1], n),
    })

    # Create perfectly correlated feature (obvious leakage)
    df["feature_leaked"] = df["label"] + np.random.randn(n) * 0.01

    results = detect_leakage(
        df,
        label_col="label",
        feature_cols=["feature_normal", "feature_leaked"],
        methods=["correlation"],
    )

    # Should detect high correlation
    assert "feature_leaked" in results["correlation"]["leaky_features"]

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

**Verify:**
```bash
pytest tests/test_leakage.py -v
```

---

### 12E-4: Test Backtester

**File (NEW):** `/Users/sneh/research/tests/test_backtest.py`

**Create:**
```python
"""
Unit tests for backtester.
"""

import pytest
import numpy as np
import pandas as pd
from src.inference.backtesting import Backtester, BacktestConfig
from src.inference.backtesting.costs import TransactionCosts

def create_test_price_data(n=1000):
    """Create synthetic price data."""
    dates = pd.date_range("2020-01-01", periods=n, freq="5min")
    df = pd.DataFrame({
        "open": 100 + np.random.randn(n).cumsum(),
        "high": 100 + np.random.randn(n).cumsum() + 1,
        "low": 100 + np.random.randn(n).cumsum() - 1,
        "close": 100 + np.random.randn(n).cumsum(),
        "volume": np.random.randint(1000, 10000, n),
    }, index=dates)

    # Ensure high >= close >= low
    df["high"] = df[["open", "high", "close"]].max(axis=1)
    df["low"] = df[["open", "low", "close"]].min(axis=1)

    return df

def test_backtest_applies_costs():
    """Verify transaction costs are applied."""
    prices = create_test_price_data(n=1000)
    predictions = pd.Series(np.random.choice([-1, 0, 1], 1000), index=prices.index)

    costs = TransactionCosts(
        commission_per_contract=2.50,
        slippage_ticks=1.0,
        tick_value=1.25,
    )

    config = BacktestConfig(
        costs=costs,
        initial_equity=100000,
    )

    backtester = Backtester(config)
    result = backtester.run(prices, predictions=predictions)

    # Total costs should be > 0
    total_costs = result.total_costs
    assert total_costs > 0, "Transaction costs not applied"

    # Net P&L should be less than gross P&L
    if result.gross_pnl != 0:
        assert result.net_pnl < result.gross_pnl, "Net P&L should be less than gross"

def test_circuit_breaker_triggers():
    """Verify circuit breaker halts trading at max drawdown."""
    prices = create_test_price_data(n=1000)

    # Create always-wrong predictions (guaranteed to lose)
    predictions = pd.Series(-np.sign(prices["close"].pct_change()), index=prices.index).fillna(0)

    config = BacktestConfig(
        max_drawdown_threshold=0.10,  # -10% halt
        initial_equity=100000,
    )

    backtester = Backtester(config)
    result = backtester.run(prices, predictions=predictions)

    # Should have stopped trading before end
    assert len(result.equity_curve) < len(prices), "Circuit breaker did not trigger"

    # Max drawdown should not exceed threshold (with small tolerance)
    max_dd = abs(result.metrics.max_drawdown)
    assert max_dd <= config.max_drawdown_threshold * 1.1, \
        f"Drawdown {max_dd:.2%} exceeded threshold {config.max_drawdown_threshold:.2%}"

def test_stop_loss_execution():
    """Verify stop losses are executed."""
    prices = create_test_price_data(n=100)
    predictions = pd.Series([1] * 100, index=prices.index)  # Always long

    config = BacktestConfig(
        use_stop_loss=True,
        stop_loss_atr_multiplier=2.0,
        initial_equity=100000,
    )

    backtester = Backtester(config)
    result = backtester.run(prices, predictions=predictions)

    # Check trades have stop_loss_price set
    trades = result.trades
    if len(trades) > 0:
        assert all(hasattr(t, "stop_loss_price") for t in trades), "Trades missing stop_loss_price"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

**Verify:**
```bash
pytest tests/test_backtest.py -v
```

---

### 12E-5: Setup pytest + fixtures

**File (NEW):** `/Users/sneh/research/tests/conftest.py`

**Create:**
```python
"""
Pytest configuration and shared fixtures.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

@pytest.fixture
def sample_ohlcv():
    """Create sample OHLCV data for testing."""
    n = 1000
    dates = pd.date_range("2020-01-01", periods=n, freq="5min")

    df = pd.DataFrame({
        "open": 100 + np.random.randn(n).cumsum(),
        "high": 100 + np.random.randn(n).cumsum() + 1,
        "low": 100 + np.random.randn(n).cumsum() - 1,
        "close": 100 + np.random.randn(n).cumsum(),
        "volume": np.random.randint(1000, 10000, n),
    }, index=dates)

    # Ensure OHLC relationships
    df["high"] = df[["open", "high", "close"]].max(axis=1)
    df["low"] = df[["open", "low", "close"]].min(axis=1)

    return df

@pytest.fixture
def sample_features(sample_ohlcv):
    """Create sample engineered features."""
    df = sample_ohlcv.copy()

    # Add some features
    df["returns"] = df["close"].pct_change()
    df["sma_20"] = df["close"].rolling(20).mean()
    df["rsi_14"] = 50 + np.random.randn(len(df)) * 10  # Simplified RSI

    # Add label
    df["label"] = np.random.choice([-1, 0, 1], len(df))

    return df.dropna()

@pytest.fixture
def temp_output_dir(tmp_path):
    """Create temporary output directory for tests."""
    output_dir = tmp_path / "test_output"
    output_dir.mkdir()
    return output_dir

@pytest.fixture
def model_contract():
    """Sample model contract."""
    from src.core.contracts import ModelContract

    return ModelContract(
        model_name="test_model",
        data_rank=2,
        input_shape=(None, 180),
        output_shape=(None, 3),
    )
```

**File (NEW):** `/Users/sneh/research/tests/__init__.py`

**Create:**
```python
"""ML Factory test suite."""
```

**File (NEW):** `/Users/sneh/research/pytest.ini`

**Create:**
```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v --tb=short --strict-markers
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks tests as integration tests
```

**Verify:**
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html
open htmlcov/index.html
```

---

## Phase 12F: Architecture Cleanup (6 tasks)

### 12F-1: Unify Exception Hierarchy

**File:** `/Users/sneh/research/src/core/exceptions.py`

**Add missing exception types (after line 100):**
```python
# Add these exception types that currently don't inherit from MLFactoryError:

class StoreError(MLFactoryError):
    """Base class for data store errors."""
    pass

class FeatureStoreError(StoreError):
    """Errors from FeatureStore operations."""
    pass

class RawMTFStoreError(StoreError):
    """Errors from RawMTFStore operations."""
    pass

class TimeframeNotFoundError(StoreError):
    """Requested timeframe not found in store."""
    pass

class SecurityError(MLFactoryError):
    """Security-related errors (suspicious data, injection attempts)."""
    pass

class NumericalError(MLFactoryError):
    """Numerical instability errors (NaN, Inf, overflow)."""
    pass

class LookaheadBiasError(ValidationError):
    """Lookahead bias detected in features or labels."""
    pass

class LeakageDetectedError(ValidationError):
    """Data leakage detected."""
    pass

class StageValidationError(ValidationError):
    """Pipeline stage validation failed."""
    pass

class ChronologicalSortError(DataError):
    """Data not in chronological order."""
    pass

class EnsembleCompatibilityError(ModelError):
    """Models incompatible for ensembling."""
    pass
```

**Update imports in affected files:**
```bash
# Update all 10+ files to use unified hierarchy
sed -i 's/class LeakageDetectedError(Exception)/class LeakageDetectedError(ValidationError)/' src/validation/leakage_detection.py
sed -i 's/from src.data.store.raw_mtf_store import RawMTFStoreError/from src.core.exceptions import RawMTFStoreError/' src/data/store/*.py
# ... (repeat for all affected files)
```

**Verify:**
```bash
# All custom exceptions should inherit from MLFactoryError
python -c "
from src.core.exceptions import MLFactoryError
from src.validation.leakage_detection import LeakageDetectedError
from src.data.store.store import FeatureStoreError
from src.validation.lookahead_audit import LookaheadBiasError

assert issubclass(LeakageDetectedError, MLFactoryError)
assert issubclass(FeatureStoreError, MLFactoryError)
assert issubclass(LookaheadBiasError, MLFactoryError)
print('✓ All exceptions inherit from MLFactoryError')
"
```

---

### 12F-2: Remove Duplicate ValidationError

**File:** `/Users/sneh/research/src/core/validation.py`

**Delete lines 40-48:**
```python
# DELETE THIS:
class ValidationError(ValueError):
    """Raised when validation fails."""
    pass
```

**Update imports (3-5 files):**
```python
# Replace:
from src.core.validation import ValidationError

# With:
from src.core.exceptions import ValidationError
```

**Files to update:**
- `src/data/pipeline/stages/validation/*.py`
- `src/validation/*.py`
- Any file that imports from `src.core.validation`

**Verify:**
```bash
# Should fail (ValidationError removed from validation.py)
python -c "from src.core.validation import ValidationError" && echo "FAIL: Still exists" || echo "OK: Removed"

# Should succeed (ValidationError in exceptions.py)
python -c "from src.core.exceptions import ValidationError" && echo "OK: Exists in exceptions.py"
```

---

### 12F-3: Delete orchestrator.py

**Files to delete:**
- `/Users/sneh/research/src/orchestrator.py` (376 lines)

**Files to update:**
```python
# /Users/sneh/research/src/__init__.py (lines 41-54)
# DELETE lazy imports for MLPipeline and PipelineResult
# Or keep with LOUD deprecation warning

def __getattr__(name: str):
    if name == "MLPipeline":
        raise ImportError(
            "MLPipeline has been removed. Use MLFactory instead:\n"
            "  from src.factory import MLFactory\n"
            "See migration guide: docs/migration_mlpipeline_to_mlfactory.md"
        )
    # ... same for PipelineResult
```

**Update CLI documentation:**
```python
# /Users/sneh/research/src/cli/commands/pipeline.py (line 6)
# UPDATE docstring to reference MLFactory instead of MLPipeline
```

**Verify:**
```bash
# Should fail with helpful error
python -c "from src import MLPipeline" && echo "FAIL" || echo "OK: Removed"

# MLFactory should still work
python -c "from src.factory import MLFactory; print('✓ MLFactory OK')"
```

---

### 12F-4: Fix Bare Exception Handlers

**27 locations to fix - Top priority files:**

**File 1:** `/Users/sneh/research/src/validation/bootstrap.py:128`
```python
# BEFORE:
except Exception:
    bootstrap_estimates[i] = np.nan

# AFTER:
except (ZeroDivisionError, FloatingPointError, ValueError) as e:
    logger.debug(f"Bootstrap sample {i} failed: {e}")
    bootstrap_estimates[i] = np.nan
```

**File 2:** `/Users/sneh/research/src/data/features/compute/wavelets.py:58,85,100`
```python
# BEFORE (appears 3 times):
except Exception:
    pass

# AFTER:
except ImportError as e:
    logger.warning(f"PyWavelets not available: {e}")
    return df  # Return without wavelet features
```

**File 3:** `/Users/sneh/research/src/factory.py:358`
```python
# BEFORE:
except Exception as e:
    logger.warning(f"Backtest failed: {e}")
    return {}

# AFTER:
except (ValueError, KeyError, BacktestError) as e:
    logger.error(f"Backtest failed: {e}", exc_info=True)
    if self.config.strict_mode:
        raise
    return {}
```

**Create checklist of all 27:**
```bash
# Find all bare exception handlers
grep -rn "except Exception:" src/ --include="*.py" > bare_exceptions.txt
# Review and fix each one
```

**Verify:**
```bash
# Should find 0 bare exceptions (or only justified ones with comments)
grep -rn "except Exception:" src/ --include="*.py" | grep -v "# Intentional broad catch"
```

---

### 12F-5: Consolidate OHLCV Validators

**File (NEW):** `/Users/sneh/research/src/core/ohlcv_validation.py`

**Create:**
```python
"""
Unified OHLCV validation.

Consolidates 4 separate validators into one with strictness levels.
"""

from enum import Enum
import pandas as pd
import numpy as np

class StrictnessLevel(Enum):
    MINIMAL = "minimal"      # Only check columns exist
    STANDARD = "standard"    # Check relationships
    STRICT = "strict"        # Check schema + data quality

def validate_ohlcv(
    df: pd.DataFrame,
    strictness: StrictnessLevel = StrictnessLevel.STANDARD,
    auto_fix: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Unified OHLCV validation.

    Args:
        df: OHLCV DataFrame
        strictness: Validation level
        auto_fix: Attempt to fix issues

    Returns:
        (validated_df, list_of_warnings)
    """
    warnings = []
    df_fixed = df.copy() if auto_fix else df

    # Level 1: Column presence (all strictness levels)
    required = ["open", "high", "low", "close", "volume"]
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Level 2: OHLC relationships (standard+)
    if strictness in [StrictnessLevel.STANDARD, StrictnessLevel.STRICT]:
        violations = (
            (df["high"] < df["close"]) |
            (df["high"] < df["open"]) |
            (df["low"] > df["close"]) |
            (df["low"] > df["open"])
        )

        if violations.any():
            n_violations = violations.sum()
            warnings.append(f"OHLC relationship violations: {n_violations} rows")

            if auto_fix:
                df_fixed.loc[violations, "high"] = df_fixed.loc[violations, [open", "high", "close"]].max(axis=1)
                df_fixed.loc[violations, "low"] = df_fixed.loc[violations, ["open", "low", "close"]].min(axis=1)

    # Level 3: Schema + data quality (strict only)
    if strictness == StrictnessLevel.STRICT:
        # Check for NaN
        if df.isna().any().any():
            warnings.append(f"Found NaN values in {df.isna().sum().sum()} cells")

        # Check for negative prices
        if (df[["open", "high", "low", "close"]] <= 0).any().any():
            warnings.append("Found non-positive prices")

        # Check for zero volume
        if (df["volume"] == 0).any():
            warnings.append(f"Found {(df['volume'] == 0).sum()} zero-volume bars")

    return df_fixed, warnings
```

**Deprecate old validators:**
```python
# Add deprecation warnings to old locations
# src/core/validation.py:336
@deprecated("Use src.core.ohlcv_validation.validate_ohlcv instead")
def validate_ohlcv(...):
    pass
```

**Verify:**
```bash
python -c "
from src.core.ohlcv_validation import validate_ohlcv, StrictnessLevel
import pandas as pd
import numpy as np

# Create test data
df = pd.DataFrame({
    'open': [100, 101, 102],
    'high': [101, 102, 103],
    'low': [99, 100, 101],
    'close': [100.5, 101.5, 102.5],
    'volume': [1000, 2000, 1500],
})

# Validate
df_valid, warnings = validate_ohlcv(df, strictness=StrictnessLevel.STRICT)
print(f'Warnings: {len(warnings)}')
"
```

---

### 12F-6: Extract Trading Constants

**File:** `/Users/sneh/research/src/models/metrics.py`

**Add at top (after imports):**
```python
# Trading Constants (extracted from magic numbers)
DEFAULT_MES_POINT_VALUE = 5.0          # Dollar value per point
DEFAULT_COMMISSION = 2.50               # Round-trip commission per contract
DEFAULT_SLIPPAGE_TICKS = 1.0           # Expected slippage in ticks
DEFAULT_MES_TICK_VALUE = 1.25          # Dollar value per tick (MES)
DEFAULT_INITIAL_EQUITY = 100_000.0     # Starting portfolio value
TRADING_DAYS_PER_YEAR = 252            # Standard market days for annualization
```

**Replace magic numbers (12 instances):**
```python
# Line 87: BEFORE
def compute_trading_metrics(trades, point_value=5.0, commission=2.50, ...):

# AFTER:
def compute_trading_metrics(
    trades,
    point_value=DEFAULT_MES_POINT_VALUE,
    commission=DEFAULT_COMMISSION,
    slippage_ticks=DEFAULT_SLIPPAGE_TICKS,
    tick_value=DEFAULT_MES_TICK_VALUE,
    initial_equity=DEFAULT_INITIAL_EQUITY,
):

# Line 342, 343, 345: Replace hardcoded 252
annual_sharpe = mean_return / (std_return + 1e-8) * np.sqrt(TRADING_DAYS_PER_YEAR)
```

**Create symbol-specific constants module:**
```python
# NEW: src/core/constants/symbols.py
MES_SPECS = {
    "point_value": 5.0,
    "tick_value": 1.25,
    "tick_size": 0.25,
    "commission": 2.50,
    "contract_multiplier": 5,  # $5 per point
}

MGC_SPECS = {
    "point_value": 10.0,
    "tick_value": 0.10,
    "tick_size": 0.10,
    "commission": 2.50,
    "contract_multiplier": 10,
}

def get_symbol_specs(symbol: str) -> dict:
    """Get contract specifications for symbol."""
    return {
        "MES": MES_SPECS,
        "MGC": MGC_SPECS,
        # ... add more
    }.get(symbol, MES_SPECS)  # Default to MES
```

**Verify:**
```bash
python -c "
from src.models.metrics import DEFAULT_MES_POINT_VALUE, TRADING_DAYS_PER_YEAR
print(f'Point value: {DEFAULT_MES_POINT_VALUE}')
print(f'Trading days: {TRADING_DAYS_PER_YEAR}')
"
```

---

## Verification Commands

```bash
# Phase 12A: Trading Profitability
python -m src optimize --metric sharpe_ratio --trials 10
python -m src backtest --realistic-execution --max-drawdown-threshold 0.10

# Phase 12B: Live Trading Safeguards
python -m src analyze-r-distribution --trades results/trades.csv
python -m src monte-carlo --trades results/trades.csv --simulations 10000
python -m src backtest --test-circuit-breaker

# Phase 12C: Deployment Infrastructure
mlflow ui &
python -m src train --symbol MES --models xgboost
curl http://localhost:8000/metrics  # Prometheus

# Phase 12D: Pipeline Performance
time python -m src pipeline --stages features  # Should use cache on 2nd run
htop  # Should see all CPUs during parallel training
nvidia-smi -l 1  # Should see GPU usage for boosting

# Phase 12E: Testing
pytest tests/ -v --cov=src --cov-report=html
open htmlcov/index.html

# Phase 12F: Architecture Cleanup
python -c "from src.core.exceptions import MLFactoryError; from src.validation.leakage_detection import LeakageDetectedError; assert issubclass(LeakageDetectedError, MLFactoryError)"
```

---

## Success Criteria

| Metric | Before | After | Status |
|--------|--------|-------|--------|
| Optuna metric | F1 score | Sharpe ratio | DONE |
| Circuit breakers | None | -10% DD, -2% daily, 5 consecutive | DONE |
| R-multiple tracking | None | Full distribution | DONE |
| Unit tests | 0 | 6 files, 981 lines | DONE |
| Training time (100 trials) | 16h | ~1.5-2h (parallel) | DONE |
| Feature cache hit | 0% | 80%+ (FeatureStore) | DONE |
| MLflow enabled | Manual | Auto-enabled | DONE |
| Stop loss enforcement | 0% | 100% (2 ATR default) | DONE |
| GPU utilization (boosting) | 0% | 80%+ (defaults enabled) | DONE |
| Exception hierarchy | Fragmented | Unified (24 classes) | DONE |

---

## Implementation Statistics

| Category | Count |
|----------|-------|
| Files Modified | ~30+ |
| Lines Added/Changed | ~5,000+ |
| New Test Files | 6 |
| New Test Lines | 981 |
| Exception Classes Consolidated | 24 |
| Performance Improvements | 5-10x+ (estimated) |

---

**Total Tasks:** 39 (37 completed, 2 skipped due to architectural mismatch)
**Actual Effort:** 7-Agent Pipeline, Single Session
**Completion Date:** 2026-01-24
**Priority Order:** 12A->12B->12E->12D->12C->12F (EXECUTED IN ORDER)

---

## PHASE 12.5: Code Quality Pass (NEW)

**Status:** NOT STARTED
**Discovered:** 2026-01-25 (Post-Phase 12 Review)
**Priority:** HIGH - Block on this before Phase 13

### Issues Discovered by 4-Agent Parallel Analysis

| Agent | Focus | Findings |
|-------|-------|----------|
| Explore | Remaining tasks | Phases 13-18 pending (32 tasks total) |
| Debugger | Tests/imports | 42 tests pass, 4 cosmetic import issues |
| Architect | Pipeline review | 5 critical architecture issues |
| Code Reviewer | Quality check | 210 ruff + 82 mypy violations |

---

### 12.5A: Ruff Auto-fixes

**Command:** `ruff check src/ --fix`

**Expected to fix ~100 violations:**
- UP038 (28): Non-PEP604 isinstance
- C401/C416 (10): Unnecessary generators
- RUF005 (10): Concatenation vs unpacking

---

### 12.5B: Ruff Unsafe Fixes (Review Required)

**Command:** `ruff check src/ --fix --unsafe-fixes`

**Review before applying:**
- SIM102 (36): Collapsible if statements
- SIM108 (27): Ternary operator conversion
- B007 (15): Unused loop variables

---

### 12.5C: Fix Critical Type Error

**File:** `/home/jake/Desktop/Research/src/core/contracts/feature_spec.py`
**Line:** 123

**Issue:** Assigns `list[Any]` to variable typed as `dict[str, Any]`

**Investigation needed** - This could be:
1. Incorrect type annotation
2. Incorrect assignment
3. Schema mismatch

**Verify:**
```bash
python -c "from src.core.contracts.feature_spec import FeatureSpec; print('OK')"
```

---

### 12.5D: Fix Silent Parallel Processing Failures

**File:** `/home/jake/Desktop/Research/src/data/pipeline/stages/features/run.py`
**Lines:** 279-286

**Current (WRONG):**
```python
for result in results:
    if result is None:  # Silently skips failures
        continue
```

**Fix:**
```python
failures = [(task, r) for task, r in zip(tasks, results) if r is None]
if failures:
    failed_symbols = [f"{s}_{tf}" for (s, tf), _ in failures]
    logger.error(f"Feature engineering failed for: {failed_symbols}")
    if config.strict_mode:
        raise StageValidationError(f"Failed: {failed_symbols}")
```

**Verify:**
```bash
# Should report failures instead of silently continuing
python -m src pipeline --stages features --strict-mode
```

---

### 12.5E: Remove Global State Mutation

**File:** `/home/jake/Desktop/Research/src/data/pipeline/stages/scaling/run.py`
**Lines:** 325-332

**Current (WRONG):**
```python
for src_file in src_scaled_dir.glob("*"):
    if src_file.is_file():
        dst_file = global_scaled_dir / src_file.name
        shutil.copy2(src_file, dst_file)  # COPIES TO SHARED DIR
```

**Issue:** Copies run-specific files to `data/splits/scaled/` which:
- Breaks run isolation
- Could overwrite data from parallel runs
- Creates hidden side effects

**Fix Options:**
1. Remove the copy entirely (use run-specific dir)
2. Make it opt-in with config flag + warning
3. Add run_id prefix to filenames

---

### 12.5F: Add Missing Stage Schemas

**File:** `/home/jake/Desktop/Research/src/data/pipeline/schemas.py`

**Missing schemas (4 of 12):**
1. `ga_optimize` - Stage 5
2. `validate_scaled` - Stage 7.6
3. `validate` - Stage 8
4. `generate_report` - Stage 9

**Add to STAGE_SCHEMAS dict:**
```python
"ga_optimize": StageSchema(
    required_columns=["label_horizon_*", "label_type"],
    optional_columns=["ga_generation", "ga_fitness"],
),
# ... etc
```

---

### 12.5G: Create StageNames Enum

**File:** `/home/jake/Desktop/Research/src/data/pipeline/stage_registry.py`

**Create enum to replace magic strings:**
```python
class StageName(str, Enum):
    INGEST = "ingest"
    CLEAN = "clean"
    FEATURES = "features"
    MTF = "mtf"
    INITIAL_LABELS = "initial_labels"
    GA_OPTIMIZE = "ga_optimize"
    FINAL_LABELS = "final_labels"
    CREATE_SPLITS = "create_splits"
    SCALING = "scaling"
    CREATE_DATASETS = "create_datasets"
    VALIDATE_SCALED = "validate_scaled"
    VALIDATE = "validate"
    GENERATE_REPORT = "generate_report"
```

**Update references in:**
- `stage_registry.py` (stage definitions)
- `runner.py` (stage_functions dict)
- `schemas.py` (STAGE_SCHEMAS dict)

---

### 12.5H: Standardize Error Handling

**Issue:** Inconsistent patterns across stages

| Stage | Pattern | Should Be |
|-------|---------|-----------|
| `run_initial_labeling` | Raises `ValueError` | Consistent |
| `_validate_horizons_vs_data` | Logs warning only | Should raise |
| `run_feature_scaling` | Raises `RuntimeError` | Consistent |
| `run_validation` | Returns failed result | Should raise |

**Recommendation:**
- All validation failures should RAISE exceptions by default
- Add `warn_only: bool = False` parameter for non-blocking mode
- Document when to use each pattern

---

### Verification Commands

```bash
# After 12.5A + 12.5B
ruff check src/  # Target: <50 violations (from 210)

# After 12.5C
mypy src/core/contracts/feature_spec.py  # No assignment error

# After 12.5D
python -m src pipeline --stages features --test-failure-handling

# After 12.5E
# Run two pipelines in parallel - should not interfere
python -m src pipeline --run-id run_001 &
python -m src pipeline --run-id run_002 &

# After all
pytest tests/ -v  # Still 42 passing
```

---

### Success Criteria

| Metric | Before | After Target |
|--------|--------|--------------|
| Ruff violations | 210 | <50 |
| Mypy critical errors | 82 | <10 |
| Silent pipeline failures | Yes | No |
| Global state mutation | Yes | No |
| Stage schemas | 8/12 | 12/12 |
| Magic strings | Many | 0 (use enum) |

---

*Last Updated: 2026-01-25 - PHASE 12.5 ADDED*
