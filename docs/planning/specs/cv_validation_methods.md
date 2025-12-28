# Advanced CV Validation Methods Specification

**Version:** 1.0.0
**Date:** 2025-12-28
**Priority:** P2 (Important)

---

## Overview

This specification covers two advanced validation methods:

1. **Lookahead Audit** - Automated verification that features don't peek into future
2. **Walk-Forward Evaluation** - Rolling-origin evaluation to detect temporal degradation

Both methods complement standard purged k-fold CV by providing additional validation guarantees.

---

## 1. Lookahead Audit

### Problem Statement

Multi-timeframe features (15min, 1H, 4H aggregations from 5min bars) may inadvertently peek into future data if resampling parameters are incorrect:

```python
# WRONG (lookahead bias):
df.resample('15T', closed='right', label='right').mean()
# This includes the current bar's close in the 15min aggregation!

# CORRECT (no lookahead):
df.resample('15T', closed='left', label='left').mean()
# Only uses completed bars
```

**Impact:** Subtle data leakage, optimistic metrics.

### Solution: Corruption Testing

Corrupt future data and verify features don't change:

```
1. Compute features for sample at index i
2. Corrupt all data after index i (set to 999999.0)
3. Recompute features for sample at index i
4. If features differ → lookahead bias detected!
```

### Implementation

**File: `src/validation/lookahead_audit.py`**

```python
class LookaheadAuditor:
    """
    Audits features for lookahead bias using corruption testing.
    
    Example:
        >>> auditor = LookaheadAuditor(feature_pipeline)
        >>> result = auditor.audit(df, n_samples=100)
        >>> assert result.passed, f"Violations: {len(result.violations)}"
    """
    
    def __init__(
        self,
        feature_func: Callable[[pd.DataFrame], pd.DataFrame],
        tolerance: float = 1e-10,
    ):
        self.feature_func = feature_func
        self.tolerance = tolerance
    
    def audit(
        self,
        df: pd.DataFrame,
        n_samples: int = 100,
    ) -> AuditResult:
        """Run lookahead audit."""
        # Compute original features
        original_features = self.feature_func(df)
        
        # Select random test indices
        test_indices = np.random.choice(
            range(100, len(df) - 100),
            size=min(n_samples, len(df) - 200),
            replace=False
        )
        
        violations = []
        
        for idx in test_indices:
            # Corrupt future data
            df_corrupted = df.copy()
            df_corrupted.iloc[idx + 1:] = 999999.0
            
            # Recompute features
            corrupted_features = self.feature_func(df_corrupted)
            
            # Compare
            orig_row = original_features.iloc[idx]
            corr_row = corrupted_features.iloc[idx]
            
            for col in original_features.columns:
                diff = abs(orig_row[col] - corr_row[col])
                if diff > self.tolerance:
                    violations.append(LookaheadViolation(
                        feature_name=col,
                        sample_index=idx,
                        original_value=orig_row[col],
                        corrupted_value=corr_row[col],
                        difference=diff,
                    ))
        
        return AuditResult(
            passed=len(violations) == 0,
            n_features_tested=len(original_features.columns),
            n_samples_tested=len(test_indices),
            violations=violations,
        )
```

### Usage

```python
from src.validation.lookahead_audit import audit_mtf_features
from src.phase1.stages.mtf.generator import MTFGenerator

# Load data
df = pd.read_parquet("data/raw/MES_1m.parquet")

# Create MTF generator
mtf_gen = MTFGenerator(...)

# Run audit
result = audit_mtf_features(df, mtf_gen)

if result.passed:
    print(f"✓ Passed: {result.n_features_tested} features, {result.n_samples_tested} samples")
else:
    print(f"✗ Failed: {len(result.violations)} violations")
    for v in result.violations[:5]:
        print(f"  {v.feature_name}: {v.original_value} → {v.corrupted_value}")
```

### Acceptance Criteria

- [ ] 0 lookahead violations detected
- [ ] All MTF features use `closed='left', label='left'`
- [ ] Audit completes in < 1 minute for 50 samples
- [ ] Clear violation reporting

---

## 2. Walk-Forward Evaluation

### Problem Statement

Purged k-fold averages performance across shuffled time periods, hiding temporal degradation:

```
K-Fold (shuffled):
  Mean F1: 0.48 ← Looks good!
  But recent performance (2024) is only 0.38 ← Hidden!

Walk-Forward (sequential):
  Window 1 (2020): F1 0.52
  Window 2 (2021): F1 0.50
  Window 3 (2022): F1 0.47
  Window 4 (2023): F1 0.44
  Window 5 (2024): F1 0.38 ← Clear degradation trend!
```

### Solution: Rolling-Origin Evaluation

Always move forward in time:

```
Expanding Window:
  Train: [2020-2021] → Test: [2022-Q1] → F1: 0.50
  Train: [2020-2022] → Test: [2022-Q2] → F1: 0.48
  Train: [2020-2022] → Test: [2023-Q1] → F1: 0.45
  ...

Rolling Window:
  Train: [2020-2021] → Test: [2022-Q1] → F1: 0.50
  Train: [2021-2022] → Test: [2022-Q2] → F1: 0.48
  Train: [2022-2023] → Test: [2023-Q1] → F1: 0.45
  ...
```

### Implementation

**File: `src/cross_validation/walk_forward.py`**

```python
@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward evaluation."""
    initial_train_size: float = 0.5      # Fraction for initial training
    step_size: int = 288                 # Bars to step forward (1 day at 5-min)
    min_test_size: int = 288             # Minimum test window
    expanding: bool = True               # True = expanding, False = rolling
    purge_bars: int = 60
    embargo_bars: int = 1440


class WalkForwardEvaluator:
    """
    Implements walk-forward (rolling-origin) evaluation.
    
    Example:
        >>> config = WalkForwardConfig(initial_train_size=0.5, step_size=288)
        >>> evaluator = WalkForwardEvaluator(config)
        >>> result = evaluator.evaluate("xgboost", X, y, horizon=20)
        >>> print(f"Mean F1: {result.mean_f1:.3f}")
        >>> print(f"Degradation: {result.temporal_degradation:.4f}")
    """
    
    def generate_windows(self, n_samples: int):
        """Generate train/test index pairs."""
        initial_train = int(n_samples * self.config.initial_train_size)
        train_end = initial_train
        
        while train_end < n_samples - self.config.min_test_size:
            # Training indices
            if self.config.expanding:
                train_start = 0
            else:
                train_start = max(0, train_end - initial_train)
            
            train_idx = np.arange(train_start, train_end - self.config.purge_bars)
            
            # Test indices (with embargo)
            test_start = train_end + self.config.embargo_bars
            test_end = min(test_start + self.config.step_size, n_samples)
            test_idx = np.arange(test_start, test_end)
            
            yield train_idx, test_idx
            
            train_end += self.config.step_size
    
    def evaluate(
        self,
        model_name: str,
        X: pd.DataFrame,
        y: pd.Series,
        horizon: int,
    ) -> WalkForwardResult:
        """Run walk-forward evaluation."""
        window_results = []
        
        for window_idx, (train_idx, test_idx) in enumerate(self.generate_windows(len(X))):
            # Train model on window
            model = ModelRegistry.create(model_name)
            model.fit(
                X_train=X.iloc[train_idx].values,
                y_train=y.iloc[train_idx].values,
                X_val=X.iloc[train_idx[-1000:]].values,
                y_val=y.iloc[train_idx[-1000:]].values,
            )
            
            # Evaluate on test
            y_pred = model.predict(X.iloc[test_idx].values).class_predictions
            y_true = y.iloc[test_idx].values
            
            window_results.append(WindowResult(
                window_idx=window_idx,
                train_size=len(train_idx),
                test_size=len(test_idx),
                accuracy=accuracy_score(y_true, y_pred),
                f1=f1_score(y_true, y_pred, average='macro'),
            ))
        
        return WalkForwardResult(
            model_name=model_name,
            horizon=horizon,
            config=self.config,
            window_results=window_results,
        )
```

### Usage

```bash
# CLI
python scripts/run_walk_forward.py --model xgboost --horizon 20 --step-size 288

# Python
from src.cross_validation.walk_forward import WalkForwardEvaluator, WalkForwardConfig

config = WalkForwardConfig(initial_train_size=0.5, step_size=288)
evaluator = WalkForwardEvaluator(config)
result = evaluator.evaluate("xgboost", X, y, horizon=20)

# Analyze results
print(f"Windows: {result.n_windows}")
print(f"Mean F1: {result.mean_f1:.3f} +/- {result.std_f1:.3f}")
print(f"Temporal degradation: {result.temporal_degradation:.4f}")

# Plot per-window performance
df = result.to_dataframe()
df.plot(x='window', y='f1', title='Walk-Forward F1 by Window')
```

### Acceptance Criteria

- [ ] Reports per-window metrics
- [ ] Shows temporal degradation curve
- [ ] Supports expanding and rolling windows
- [ ] Handles edge cases (insufficient data)
- [ ] Temporal ordering preserved

---

## Cross-References

- [ROADMAP.md](../ROADMAP.md) - Implementation timeline
- [GAPS_ANALYSIS.md](../GAPS_ANALYSIS.md) - Gap details
- [specs/cv_improvements.md](cv_improvements.md) - Core CV fixes

---

## Document Control

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-12-28 | ML Engineering | Initial spec from IMPLEMENTATION_PLAN.md |
