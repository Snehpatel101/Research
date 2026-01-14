# MLOps Quick Start Guide

**Quick reference for using MLOps components in ML_Pipeline.ipynb**

## 🚀 Quick Start

### 1. Enable MLOps Features (Cell 1.3)

```python
# Essential gates - always enable in production
ENABLE_DATA_QUALITY_GATES = True
ENABLE_DRIFT_DETECTION = True
ENABLE_PERFORMANCE_MONITORING = True
USE_DEPLOYMENT_GATES = True
```

### 2. Run Pipeline with Gates

```python
# Standard workflow:
# 1. Cell 3.2: Run Data Pipeline
# 2. Cell 3.6: Data Quality Validation ⚡ AUTO-BLOCKS if quality low
# 3. Cell 4.1: Train Models
# 4. Cell 4.6: Performance Monitoring ⚡ AUTO-ALERTS if degradation
# 5. Cell 5.6: Drift Detection ⚡ AUTO-RECOMMENDS retrain if drift
# 6. Cell 5.7: Deployment Gate ⚡ AUTO-BLOCKS if criteria not met
```

### 3. Check Deployment Status

```python
# After Cell 5.7
if CONFIG.deployment_gate_passed:
    print("✅ APPROVED - Deploy to production")
else:
    print("❌ BLOCKED - Check CONFIG.deployment_gate_report")
    print(CONFIG.deployment_gate_report)
```

## 🎯 Key Thresholds

| Component | Parameter | Default | Recommendation |
|-----------|-----------|---------|----------------|
| **Data Quality** | `MIN_DATA_QUALITY_SCORE` | 0.8 | Keep at 0.8 |
| **Drift Detection** | `DRIFT_THRESHOLD` | 0.05 | 0.05 (KS test) |
| **Performance** | `ALERT_THRESHOLD_ACCURACY` | 0.05 | 5% drop = alert |
| **Sharpe Gate** | `GATE_MIN_SHARPE` | 1.0 | Start at 1.0 |
| **Drawdown Gate** | `GATE_MAX_DRAWDOWN` | 0.30 | Max 30% |
| **PBO Gate** | `GATE_MAX_PBO` | 0.50 | Industry standard |

## 📊 Interpreting Results

### Data Quality Score

```python
# Cell 3.6 output
quality_score = 0.85  # Example

if quality_score >= 0.9:
    # ✅ Excellent - proceed
elif quality_score >= 0.8:
    # ✅ Good - proceed with caution
elif quality_score >= 0.7:
    # ⚠️ Fair - investigate issues
else:
    # ❌ Poor - blocked
```

**Components:**
- Missing values: 30% weight
- Outliers: 20% weight
- Correlation: 20% weight
- Balance: 15% weight
- Freshness: 10% weight
- Range: 5% weight

### Drift Detection

```python
# Cell 5.6 output
drift_pct = 25%  # Example: 25% of features drifted

if drift_pct > 30:
    # 🔴 HIGH DRIFT - retrain immediately
elif drift_pct > 15:
    # 🟡 MODERATE DRIFT - monitor closely
else:
    # 🟢 LOW DRIFT - continue monitoring
```

**PSI Interpretation (if using PSI method):**
- PSI < 0.1: No significant drift
- PSI 0.1-0.25: Moderate drift
- PSI > 0.25: Significant drift

### Performance Degradation

```python
# Cell 4.6 output
accuracy_drop = 0.08  # Example: 8% drop

if accuracy_drop > ALERT_THRESHOLD_ACCURACY:
    # 🔴 ALERT - retrain recommended
elif accuracy_drop > ALERT_THRESHOLD_ACCURACY / 2:
    # 🟡 WARNING - monitor
else:
    # ✅ OK - no degradation
```

### Deployment Gate

```python
# Cell 5.7 output
gate_report = {
    'sharpe': {'value': 1.2, 'threshold': 1.0, 'passed': True},
    'max_drawdown': {'value': 0.25, 'threshold': 0.30, 'passed': True},
    'pbo': {'value': 0.45, 'threshold': 0.50, 'passed': True},
    'data_quality': {'value': 0.85, 'threshold': 0.80, 'passed': True},
    'calibration': {'value': 0.08, 'threshold': 0.10, 'passed': True},
    'degradation': {'value': 'No', 'threshold': 'No', 'passed': True}
}

# ALL must pass → deployment approved ✅
```

## 🔧 Common Scenarios

### Scenario 1: Data Quality Gate Fails

```python
# Cell 3.6 output:
# ❌ [BLOCKED] Data quality too low: 0.72 < 0.80

# Actions:
1. Check CONFIG.quality_checks for failed components
2. Investigate specific issues:
   - High missing values → impute or drop
   - Too many outliers → winsorize or cap
   - High correlation → drop redundant features
   - Class imbalance → resample or weight
3. Fix issues and re-run Cell 3.2 (pipeline)
4. Re-run Cell 3.6 (validation)
```

### Scenario 2: High Drift Detected

```python
# Cell 5.6 output:
# ⚠️ DRIFT DETECTED in 45 features (30%)
# 🔴 HIGH DRIFT: Consider retraining

# Actions:
1. Check CONFIG.drift_results['drifted_features']
2. Investigate top drifted features
3. Options:
   a. Retrain on recent data (recommended)
   b. Drop drifted features (if noise)
   c. Add regime indicators (if market shift)
4. Re-run entire pipeline with new data
```

### Scenario 3: Performance Degradation

```python
# Cell 4.6 output:
# 🔴 [ALERT] Performance degradation detected!
# Accuracy dropped by 0.07 (>0.05 threshold)

# Actions:
1. Check if drift detected (Cell 5.6)
2. Check regime change (Cell 3.4)
3. Options:
   a. Retrain on recent data
   b. Adjust features (remove drifted)
   c. Change model architecture
4. Re-validate with Cell 5.7
```

### Scenario 4: Deployment Gate Blocked

```python
# Cell 5.7 output:
# ❌ DEPLOYMENT BLOCKED
# Failed gates:
#   • Sharpe Ratio: 0.8 (required: >= 1.0)
#   • PBO: 0.65 (required: <= 0.50)

# Actions for Sharpe < 1.0:
1. Tune hyperparameters (Cell 4.1)
2. Add more features (Cell 3.2)
3. Try different model (Cell 4.1)
4. Adjust transaction costs (Cell 1.2)

# Actions for PBO > 0.50:
1. Model is overfit - simplify
2. Use walk-forward validation (Cell 3.5)
3. Reduce feature count
4. Increase regularization
```

## 🎛️ Tuning Guide

### Conservative (Strict Gates)

```python
# Use in production - strict quality control
MIN_DATA_QUALITY_SCORE = 0.85
DRIFT_THRESHOLD = 0.01  # Very sensitive
ALERT_THRESHOLD_ACCURACY = 0.03  # Alert on 3% drop
GATE_MIN_SHARPE = 1.5
GATE_MAX_DRAWDOWN = 0.20
GATE_MAX_PBO = 0.40
```

### Standard (Recommended)

```python
# Default - balanced quality vs flexibility
MIN_DATA_QUALITY_SCORE = 0.80
DRIFT_THRESHOLD = 0.05
ALERT_THRESHOLD_ACCURACY = 0.05
GATE_MIN_SHARPE = 1.0
GATE_MAX_DRAWDOWN = 0.30
GATE_MAX_PBO = 0.50
```

### Relaxed (Development)

```python
# Use in research - looser gates
MIN_DATA_QUALITY_SCORE = 0.70
DRIFT_THRESHOLD = 0.10
ALERT_THRESHOLD_ACCURACY = 0.10
GATE_MIN_SHARPE = 0.5
GATE_MAX_DRAWDOWN = 0.40
GATE_MAX_PBO = 0.70
```

## 📈 Monitoring Checklist

### Pre-Training
- [ ] Data quality score >= 0.8
- [ ] No critical data issues (missing, outliers)
- [ ] Class balance acceptable

### Post-Training
- [ ] Performance monitoring shows no degradation
- [ ] Drift detection shows <30% drift
- [ ] All deployment gates pass

### Production
- [ ] Monitor drift weekly/monthly
- [ ] Re-run Cell 4.6 on new data
- [ ] Retrain if any gate fails

## 🆘 Troubleshooting

### Issue: Cell 3.6 fails with "CONFIG.train_data not found"

**Solution:**
```python
# Run Cell 3.2 first (pipeline must complete)
# Then run Cell 3.6
```

### Issue: Cell 4.6 fails with "best_model_name not found"

**Solution:**
```python
# Run Cell 4.1 first (train models)
# Then run Cell 4.6
```

### Issue: Cell 5.6 shows 100% drift

**Solution:**
```python
# Likely data issue - check:
1. Train/test splits are correct
2. Scaling applied only on train
3. No data leakage
```

### Issue: All deployment gates fail

**Solution:**
```python
# Check each component:
1. Cell 4.5 (trading sim) - are metrics calculated?
2. Cell 5.5 (CPCV+PBO) - is PBO calculated?
3. Cell 3.6 (quality) - is quality_score stored?
4. Cell 4.5 (calibration) - is ECE stored?
```

## 💡 Pro Tips

### Tip 1: Save MLOps Reports
```python
# After Cell 5.7
import json
mlops_report = {
    'quality': CONFIG.data_quality_score,
    'drift': CONFIG.drift_results,
    'performance': CONFIG.performance_monitoring,
    'deployment': CONFIG.deployment_gate_report
}

with open('mlops_report.json', 'w') as f:
    json.dump(mlops_report, f, indent=2)
```

### Tip 2: Compare Across Runs
```python
# Track quality/drift over time
history = []
history.append({
    'date': '2026-01-12',
    'quality': CONFIG.data_quality_score,
    'drift_pct': CONFIG.drift_results['drift_pct'],
    'sharpe': CONFIG.trading_sim_results['sharpe_ratio']
})
```

### Tip 3: Automated Alerts
```python
# Add to Cell 5.7
if not CONFIG.deployment_gate_passed:
    # Send alert (email, Slack, etc.)
    send_alert(f"Deployment blocked: {CONFIG.deployment_gate_report}")
```

### Tip 4: Regime-Aware Monitoring
```python
# Check if degradation is regime-specific
if CONFIG.performance_monitoring['degradation_detected']:
    # Compare current regime vs training regimes
    current_regime = detect_regime(recent_data)
    if current_regime not in training_regimes:
        print("Degradation due to new regime - retrain recommended")
```

## 📚 Further Reading

- **Data Quality**: `/docs/MLOPS_INTEGRATION.md` - Section "Data Quality Validation"
- **Drift Detection**: `/docs/MLOPS_INTEGRATION.md` - Section "Feature Drift Detection"
- **Deployment Gates**: `/docs/MLOPS_INTEGRATION.md` - Section "Deployment Decision Gate"
- **Architecture**: `/docs/ARCHITECTURE.md` - Overall pipeline architecture

## 🔗 Related Cells

| Cell | Purpose | MLOps Integration |
|------|---------|-------------------|
| 1.3 | MLOps Config | Configure all gates |
| 2.4 | Helper Functions | Utility functions |
| 3.6 | Data Quality | GATE #1 (blocks pipeline) |
| 4.6 | Performance Monitor | Alert system |
| 5.6 | Drift Detection | Retrain recommendations |
| 5.7 | Deployment Gate | GATE #2 (blocks deployment) |

---

**Quick Reference Card**

```
┌─────────────────────────────────────────────────────────────┐
│ MLOps Pipeline Flow                                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Configure (Cell 1.3)                                    │
│     └─ Enable gates, set thresholds                         │
│                                                             │
│  2. Data Pipeline (Cell 3.2)                                │
│     └─ Process data                                         │
│                                                             │
│  3. Quality Gate (Cell 3.6) ⚡                              │
│     └─ BLOCKS if quality < 0.8                              │
│                                                             │
│  4. Train (Cell 4.1)                                        │
│     └─ Train models                                         │
│                                                             │
│  5. Performance Monitor (Cell 4.6) ⚡                       │
│     └─ ALERTS if degradation > 5%                           │
│                                                             │
│  6. Drift Detection (Cell 5.6) ⚡                           │
│     └─ RECOMMENDS retrain if drift > 30%                    │
│                                                             │
│  7. Deployment Gate (Cell 5.7) ⚡                           │
│     └─ BLOCKS if any criteria fail                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Version:** 1.0
**Last Updated:** 2026-01-12
