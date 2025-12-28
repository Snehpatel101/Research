# Combinatorial Purged Cross-Validation Specification

**Version:** 1.0.0
**Date:** 2025-12-28
**Priority:** P2 (Robustness)

---

## Overview

### Problem Statement

Standard k-fold CV tests a single temporal path through the data. With hyperparameter tuning across 100+ trials, the winner may overfit to **that specific path**.

**Example:**
```
Standard 5-fold CV (single path):
  Fold splits: [A,B,C,D,E]
  Trial 1 (params_1): Test on [E], train on [A,B,C,D] → F1: 0.50
  Trial 2 (params_2): Test on [E], train on [A,B,C,D] → F1: 0.52 ← Winner!
  
But what if params_2 only works well on path [A,B,C,D]→[E]?
```

### Solution: CPCV (Combinatorial Purged CV)

Test **multiple** time-group combinations to estimate robustness:

```
CPCV with 6 groups, test on 2:
  C(6,2) = 15 combinations
  
  Combination 1: Test [A,B], Train [C,D,E,F]
  Combination 2: Test [A,C], Train [B,D,E,F]
  ...
  Combination 15: Test [E,F], Train [A,B,C,D]
  
For each hyperparameter config, get 15 scores instead of 1.
Compute PBO (Probability of Backtest Overfitting).
```

### PBO (Probability of Backtest Overfitting)

```
PBO = P(IS-best strategy performs OOS-below-median)

Interpretation:
  PBO < 0.3: Low overfitting risk
  PBO 0.3-0.5: Moderate risk
  PBO > 0.5: High risk (model selection unreliable)
  PBO > 0.8: Severe overfitting (block deployment)
```

**Reference:** Bailey et al. (2014) "The Probability of Backtest Overfitting"

---

## Implementation

### File: `src/cross_validation/cpcv.py`

```python
@dataclass
class CPCVConfig:
    """Configuration for CPCV."""
    n_groups: int = 6                    # Number of time groups
    n_test_groups: int = 2               # Groups held out for test
    purge_bars: int = 60
    embargo_bars: int = 1440
    max_combinations: int = 15           # Limit for computation


class CombinatorialPurgedCV:
    """
    Combinatorial Purged Cross-Validation.
    
    Tests C(n,k) combinations of time groups to assess robustness.
    
    Example:
        >>> cpcv = CombinatorialPurgedCV(CPCVConfig(n_groups=6, n_test_groups=2))
        >>> for train_idx, test_idx in cpcv.split(X):
        ...     # Train and evaluate
        >>> pbo = compute_pbo(oos_scores, is_scores)
        >>> print(f"PBO: {pbo:.3f}")
    """
    
    def __init__(self, config: CPCVConfig):
        self.config = config
        
        # Generate all combinations
        from itertools import combinations
        self._combinations = list(combinations(
            range(config.n_groups),
            config.n_test_groups
        ))
        
        # Limit if too many
        if len(self._combinations) > config.max_combinations:
            np.random.shuffle(self._combinations)
            self._combinations = self._combinations[:config.max_combinations]
    
    def split(self, X: pd.DataFrame, y=None):
        """Generate train/test splits for all combinations."""
        n_samples = len(X)
        group_size = n_samples // self.config.n_groups
        
        # Create time groups
        groups = []
        for i in range(self.config.n_groups):
            start = i * group_size
            end = (i + 1) * group_size if i < self.config.n_groups - 1 else n_samples
            groups.append(np.arange(start, end))
        
        for test_group_indices in self._combinations:
            # Test indices
            test_idx = np.concatenate([groups[i] for i in test_group_indices])
            
            # Training indices (with purge/embargo)
            train_mask = np.ones(n_samples, dtype=bool)
            train_mask[test_idx] = False
            
            # Apply purge and embargo
            for test_group in test_group_indices:
                test_start = groups[test_group][0]
                purge_start = max(0, test_start - self.config.purge_bars)
                train_mask[purge_start:test_start] = False
                
                test_end = groups[test_group][-1] + 1
                embargo_end = min(n_samples, test_end + self.config.embargo_bars)
                train_mask[test_end:embargo_end] = False
            
            train_idx = np.where(train_mask)[0]
            
            yield train_idx, test_idx


def compute_pbo(
    oos_scores: np.ndarray,
    is_scores: np.ndarray,
) -> float:
    """
    Compute Probability of Backtest Overfitting.
    
    Args:
        oos_scores: Out-of-sample scores for each combination
        is_scores: In-sample scores for each combination
    
    Returns:
        PBO estimate (0 = no overfitting, 1 = complete overfitting)
    """
    n = len(oos_scores)
    if n == 0:
        return 0.5
    
    # Rank strategies by IS performance
    is_ranks = np.argsort(np.argsort(is_scores))[::-1]
    oos_ranks = np.argsort(np.argsort(oos_scores))[::-1]
    
    # PBO = probability that IS-best is OOS-worst-half
    best_is_idx = np.argmax(is_scores)
    pbo = float(oos_ranks[best_is_idx] > n // 2)
    
    return pbo
```

---

## Usage

### Hyperparameter Tuning with CPCV

```python
from src.cross_validation.cpcv import CombinatorialPurgedCV, compute_pbo

# Setup CPCV
cpcv = CombinatorialPurgedCV(CPCVConfig(n_groups=6, n_test_groups=2))

# Run Optuna with CPCV
def objective(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
    }
    
    scores = []
    for train_idx, test_idx in cpcv.split(X):
        model = XGBoostModel(params)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        score = model.evaluate(X.iloc[test_idx], y.iloc[test_idx])
        scores.append(score)
    
    return np.mean(scores)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

# Compute PBO
oos_scores = []
is_scores = []
for trial in study.trials:
    # Split trial scores into IS and OOS
    # (implementation depends on how you track this)
    pass

pbo = compute_pbo(oos_scores, is_scores)
print(f"PBO: {pbo:.3f}")

if pbo > 0.5:
    print("⚠️  WARNING: High overfitting risk detected!")
if pbo > 0.8:
    print("🛑 CRITICAL: Severe overfitting! Block deployment.")
```

---

## Acceptance Criteria

- [ ] PBO estimate computed after hyperparameter tuning
- [ ] Warning when PBO > 0.5
- [ ] Block deployment when PBO > 0.8
- [ ] Supports configurable n_groups and n_test_groups
- [ ] Purge/embargo applied correctly
- [ ] Computation completes in reasonable time (< 5min)

---

## Cross-References

- [ROADMAP.md](../ROADMAP.md#24-cpcv-combinatorial-purged-cv) - Phase 2 overview
- [GAPS_ANALYSIS.md](../GAPS_ANALYSIS.md#gap-9-no-cpcv-p2) - Gap details

---

## Document Control

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-12-28 | ML Engineering | Initial CPCV spec from IMPLEMENTATION_PLAN.md |
