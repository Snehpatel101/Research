# PHASE 4: META-LEARNERS - Implementation Plan

**Status:** ✅ COMPLETE (95%)
**Last Updated:** 2026-01-18
**Dependencies:** PHASE_0, PHASE_2, PHASE_3

---

## Executive Summary

PHASE_4 establishes the meta-learner system for building heterogeneous ensembles. The key challenge is **OOF alignment** - combining tabular (100% coverage) with sequence (~98% coverage) models.

---

## Current State Analysis

### Package Structure

```
src/models/ensemble/
├── __init__.py                  ✅ Complete - All exports
├── voting.py                    ✅ Complete - VotingEnsemble
├── stacking.py                  ✅ Complete - StackingEnsemble
├── blending.py                  ✅ Complete - BlendingEnsemble
├── heterogeneous_stacking.py    ✅ Complete - HeterogeneousStackingBuilder
├── meta_factory.py              ✅ Complete - MetaLearnerFactory
├── orchestrator.py              ✅ Complete - EnsembleOrchestrator
├── ridge_meta.py                ✅ Complete - RidgeMetaLearner
├── mlp_meta.py                  ✅ Complete - MLPMetaLearner
├── xgboost_meta.py              ✅ Complete - XGBoostMeta
├── calibrated_meta.py           ✅ Complete - CalibratedMetaLearner
├── diversity.py                 ✅ Complete - DiversityAnalyzer
└── validator.py                 ✅ Complete - Ensemble validators
```

---

## Implemented Components

### 1. Meta-Learner Factory (`meta_factory.py`)

```python
# Key exports:
MetaLearnerFactory        # Config-driven creation
MetaLearnerConfig         # Configuration dataclass
META_LEARNER_REGISTRY     # Registry of available meta-learners
get_meta_learner          # Factory function
create_meta_learner_from_config  # Config-based creation
list_meta_learners        # List available

# Usage:
factory = MetaLearnerFactory(config)
meta = factory.create("ridge_meta")

# Or directly:
meta = get_meta_learner("xgboost_meta", n_estimators=100)
```

### 2. Meta-Learners (4 Types)

| Meta-Learner | Description | Use Case |
|--------------|-------------|----------|
| `ridge_meta` | L2-regularized logistic | Fast baseline |
| `mlp_meta` | 2-layer MLP | Non-linear interactions |
| `xgboost_meta` | XGBoost with calibration | High diversity bases |
| `calibrated_meta` | Isotonic calibration | Probability calibration |

### 3. HeterogeneousStackingBuilder (`heterogeneous_stacking.py`)

```python
# Key exports:
HeterogeneousStackingBuilder  # Builds aligned stacking datasets
StackingFeatures              # Result dataclass
build_stacking_features       # Convenience function

# Handles OOF alignment:
# - Tabular (XGBoost): 100% coverage, offset=0
# - Sequence (LSTM): ~98% coverage, offset=seq_len-1

builder = HeterogeneousStackingBuilder(config)
features = builder.build(
    oof_predictions={"xgb": oof_xgb, "lstm": oof_lstm},
    y_true=labels,
)

# features.X: Probability features + derived (mean_conf, agreement, entropy)
# features.y: Aligned labels
# features.coverage: Coverage ratio
```

### 4. EnsembleOrchestrator (`orchestrator.py`)

**THE single entry point for ensemble training.**

```python
# Key exports:
EnsembleOrchestrator  # Master controller
EnsembleResult        # Complete ensemble output
build_ensemble        # Convenience function

# Usage:
orchestrator = EnsembleOrchestrator(config)
result = orchestrator.train(
    oof_predictions={"xgb": oof_xgb, "lstm": oof_lstm},
    y_train=labels,
)

# result contains:
#   - ensemble_name: Identifier
#   - meta_learner_name: Type used
#   - base_model_names: List of bases
#   - metrics: Ensemble metrics
#   - coverage: Sample coverage
#   - alignment_offset: OOF alignment offset
```

### 5. Diversity Analysis (`diversity.py`)

```python
# Key exports:
DiversityAnalyzer   # Analyze ensemble diversity
DiversityMetrics    # Metrics dataclass

# Metrics computed:
# - pairwise_correlation: How similar are predictions
# - disagreement: How often models disagree
# - q_statistic: Independence measure
# - diversity_score: Overall diversity

analyzer = DiversityAnalyzer()
metrics = analyzer.analyze(oof_predictions)
print(f"Diversity score: {metrics.diversity_score}")
```

---

## OOF Alignment Strategy

```
XGBoost OOF (2D)          LSTM OOF (3D)
Coverage: 100%            Coverage: 98%
Offset: 0                 Offset: 59 (seq_len-1)
        │                       │
        ▼                       ▼
    ┌───────────────────────────────┐
    │     OOF ALIGNMENT             │
    │  max_offset = 59              │
    │  common_samples = N - 59      │
    └───────────────────────────────┘
                │
                ▼
    ┌───────────────────────────────┐
    │  STACKING FEATURES            │
    │  - xgb_prob_short/neutral/long │
    │  - lstm_prob_short/neutral/long│
    │  - mean_confidence (derived)  │
    │  - prediction_agreement       │
    │  - prediction_entropy         │
    └───────────────────────────────┘
                │
                ▼
        META-LEARNER
```

---

## Remaining Tasks

### Task 4.1: Add OOFCache ⚠️

**Gap:** No caching of OOF predictions for efficient re-use.

**Action Items:**
- [ ] Implement `src/ensemble/oof_cache.py`
- [ ] Hash-based cache keys from config
- [ ] Cache invalidation when data changes

### Task 4.2: Validate Integration with PHASE_3 ⚠️

**Gap:** Need end-to-end test from training to ensemble.

**Action Items:**
- [ ] Integration test: TrainingRunResult → EnsembleOrchestrator
- [ ] Validate OOF format compatibility
- [ ] Test with actual LSTM/XGBoost predictions

---

## Usage Examples

### Example 1: Build Ensemble from Training Result
```python
from src.training import UnifiedTrainingOrchestrator
from src.models.ensemble import EnsembleOrchestrator

# Train base models
train_config = PipelineConfig(
    models=["xgboost", "lightgbm", "lstm"],
    save_oof=True,
    build_ensemble=False,  # We'll build manually
)
training_result = UnifiedTrainingOrchestrator(train_config).train(df)

# Build ensemble
ens_orchestrator = EnsembleOrchestrator(config)
ensemble_result = ens_orchestrator.train_from_training_result(training_result)

print(f"Coverage: {ensemble_result.coverage:.2%}")
print(f"Metrics: {ensemble_result.metrics}")
```

### Example 2: Analyze Diversity
```python
from src.models.ensemble import DiversityAnalyzer

analyzer = DiversityAnalyzer()
metrics = analyzer.analyze(
    oof_predictions={"xgb": oof_xgb, "lgb": oof_lgb, "lstm": oof_lstm}
)

print(f"Diversity score: {metrics.diversity_score:.4f}")
print(f"Mean agreement: {1 - metrics.disagreement:.2%}")
print(f"Recommendations: {metrics.recommendations}")
```

---

## Sign-off Criteria

- [x] MetaLearnerFactory with 4 meta-learners
- [x] HeterogeneousStackingBuilder with alignment
- [x] EnsembleOrchestrator as entry point
- [x] DiversityAnalyzer for ensemble analysis
- [x] StackingFeatures with derived features
- [ ] OOFCache implementation
- [ ] End-to-end integration test

**PHASE_4 Status: READY FOR PHASE_5**
