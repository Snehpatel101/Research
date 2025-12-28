# Online Drift Detection Specification

**Version:** 1.0.0
**Date:** 2025-12-28
**Priority:** P1 (Production Safety)

---

## Overview

### Problem Statement

Models decay silently in production without monitoring. The current pipeline has offline PSI checks in Phase 1 validation, but no runtime monitoring for:
- Market regime changes (e.g., low vol → high vol)
- Feature distribution shifts
- Performance degradation

**Impact:** Silent model failure, capital loss, delayed detection (days/weeks).

### Solution

Real-time drift monitoring using:
1. **ADWIN** (ADaptive WINdowing) - Performance/concept drift
2. **PSI** (Population Stability Index) - Feature distribution drift
3. **Alert System** - Configurable thresholds and handlers

### Dependencies

```bash
pip install river>=0.21.0
```

---

## Technical Design

### Drift Types

| Type | Detection Method | Threshold | Action |
|------|------------------|-----------|--------|
| **Feature Drift** | PSI per feature | PSI > 0.2 | Critical alert, retrain |
| **Concept Drift** | ADWIN on accuracy | Performance drop > 25% | Critical alert, halt trading |
| **Prediction Drift** | PSI on predictions | PSI > 0.15 | Warning, monitor |

### PSI (Population Stability Index)

```python
PSI = sum((actual_% - expected_%) * log(actual_% / expected_%))

Interpretation:
  < 0.1: No drift
  0.1-0.2: Moderate drift (warning)
  > 0.2: Significant drift (critical)
```

### ADWIN (Adaptive Windowing)

- Maintains sliding window of recent predictions
- Splits window at change points
- Detects shifts within ~100 samples

---

## Implementation Outline

### File Structure

```
src/monitoring/
    __init__.py
    drift_detector.py       # DriftDetector, ADWIN, PSI
    performance_monitor.py  # PerformanceMonitor
    alert_handlers.py       # Slack, email, logging
```

### Core Classes

**`ADWINDetector`:**
- Wraps `river.drift.ADWIN`
- Monitors correctness stream (1 = correct, 0 = wrong)
- Returns True when drift detected

**`PSICalculator`:**
- Fits reference distribution (training data)
- Computes PSI between reference and current batch
- Returns PSI score

**`OnlineDriftMonitor`:**
- Combines ADWIN + PSI
- Manages alert callbacks
- Tracks drift history

### Usage Example

```python
from src.monitoring.drift_detector import OnlineDriftMonitor, DriftConfig

# Initialize with reference data
monitor = OnlineDriftMonitor(
    reference_data=X_train[:5000].values,
    config=DriftConfig(
        psi_warning_threshold=0.1,
        psi_critical_threshold=0.2,
        adwin_delta=0.002,
    ),
    feature_names=feature_columns,
    alert_callback=send_slack_alert,
)

# In production loop
for batch in production_data:
    alerts = monitor.check(
        features=batch[feature_columns].values,
        predictions=model.predict(batch),
        actuals=batch['label'] if available else None,
    )
    
    for alert in alerts:
        if alert.severity == AlertSeverity.CRITICAL:
            # Halt trading, retrain model
            handle_critical_drift(alert)
```

---

## Full Implementation

**File: `src/monitoring/drift_detector.py`**

See IMPLEMENTATION_PLAN.md section 6.4 for complete 358-line implementation including:
- `DriftType` and `AlertSeverity` enums
- `DriftAlert` and `DriftConfig` dataclasses
- `ADWINDetector` class (~60 lines)
- `PSICalculator` class (~50 lines)
- `OnlineDriftMonitor` class (~200 lines)
- Helper functions

Key methods:
- `ADWINDetector.update(value)` → bool (drift detected)
- `PSICalculator.fit_reference(values)` → None
- `PSICalculator.compute_psi(values)` → float
- `OnlineDriftMonitor.check(features, predictions, actuals)` → List[DriftAlert]

---

## Testing

### Unit Tests

```python
def test_adwin_detects_shift():
    """ADWIN should detect mean shift."""
    detector = ADWINDetector(delta=0.002)
    
    # Feed stable data
    for _ in range(100):
        detector.update(0.8)  # 80% accuracy
    
    # Inject drift
    detected = False
    for _ in range(100):
        if detector.update(0.5):  # Drop to 50%
            detected = True
            break
    
    assert detected, "Should detect accuracy drop"


def test_psi_computes_correctly():
    """PSI should match expected values."""
    calc = PSICalculator(n_bins=10)
    
    # Reference: normal distribution
    ref = np.random.normal(0, 1, 10000)
    calc.fit_reference(ref)
    
    # Same distribution: PSI ~0
    current_same = np.random.normal(0, 1, 1000)
    psi_same = calc.compute_psi(current_same)
    assert psi_same < 0.05, f"PSI should be low for same dist, got {psi_same}"
    
    # Shifted distribution: PSI > 0.2
    current_shifted = np.random.normal(1, 1, 1000)
    psi_shifted = calc.compute_psi(current_shifted)
    assert psi_shifted > 0.2, f"PSI should be high for shifted dist, got {psi_shifted}"
```

### Integration Test

```python
# Create monitor
monitor = OnlineDriftMonitor(
    reference_data=X_train.values,
    config=DriftConfig(),
    feature_names=list(X_train.columns),
)

# Simulate production batches
for i in range(10):
    # Normal batch
    batch = X_test.iloc[i*100:(i+1)*100].values
    alerts = monitor.check(batch)
    assert len(alerts) == 0, "No drift expected in test data"

# Inject drifted batch
drifted_batch = X_test.iloc[:100].values * 2.0
alerts = monitor.check(drifted_batch)
assert len(alerts) > 0, "Should detect drift"
assert any(a.drift_type == DriftType.FEATURE for a in alerts)
```

---

## Acceptance Criteria

- [ ] ADWIN detects performance drift within 100 samples
- [ ] PSI alerts when feature drift > 0.2
- [ ] Alert callbacks fire correctly
- [ ] Configurable thresholds work
- [ ] No false positives on stable data
- [ ] Integration with inference pipeline complete

---

## Cross-References

- [ROADMAP.md](../ROADMAP.md#21-online-drift-detection) - Phase 2 overview
- [GAPS_ANALYSIS.md](../GAPS_ANALYSIS.md#gap-4-no-online-drift-detection) - Gap details
- [specs/inference_pipeline.md](inference_pipeline.md) - Integration with deployment

---

## Document Control

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-12-28 | ML Engineering | Initial drift detection spec from IMPLEMENTATION_PLAN.md |
