# Phase 8: Meta-Learners and Adaptive Ensembles

**Status:** 📋 Planned (not yet implemented)
**Effort:** 5-7 days (estimated)
**Dependencies:** Phase 7 (ensemble models)

---

## Goal

Build second-level meta-learners that adaptively combine ensemble outputs based on market regimes, recent performance, and prediction confidence to maximize robustness and performance across changing market conditions.

**Output:** Adaptive meta-learner models that dynamically weight ensemble predictions.

---

## Motivation

**Problem with Static Ensembles:**
- Fixed weights don't adapt to changing market conditions
- Some models perform better in trending vs ranging markets
- Ensemble diversity valuable but weighting is static

**Meta-Learner Solution:**
- Learn to weight ensembles based on context (regime, volatility, etc.)
- Adapt weights based on recent performance
- Confidence-based weighting (downweight uncertain predictions)

---

## Architecture: Two-Level Ensemble

```
Base Models (Phase 6)
  ├→ XGBoost
  ├→ LightGBM
  ├→ CatBoost
  ├→ LSTM
  ├→ GRU
  └→ TCN
       ↓
Ensemble Models (Phase 7)
  ├→ Voting (Boosting)
  ├→ Stacking (Boosting)
  ├→ Voting (Neural)
  └→ Stacking (Neural)
       ↓
Meta-Learner (Phase 8)
  ├→ Regime-Aware Weighting
  ├→ Confidence-Based Selection
  └→ Adaptive Ensemble Combination
       ↓
Final Predictions
```

---

## Data Contracts

### Input: Ensemble Predictions

**From Phase 7:**
```python
{
    "ensemble_predictions": {
        "voting_boosting": PredictionOutput,
        "stacking_boosting": PredictionOutput,
        "voting_neural": PredictionOutput,
        "stacking_neural": PredictionOutput
    },
    "base_predictions": {
        "xgboost": PredictionOutput,
        "lightgbm": PredictionOutput,
        # ... all base models
    },
    "market_context": {
        "regime": str,  # 'trending', 'ranging', 'volatile'
        "volatility": float,
        "recent_performance": Dict[str, float]  # Per-model accuracy last N bars
    }
}
```

### Output: Meta-Learner Predictions

```python
@dataclass
class MetaLearnerOutput:
    predictions: np.ndarray         # Final predictions
    probabilities: np.ndarray       # Final probabilities
    confidence: np.ndarray          # Confidence scores
    ensemble_weights: np.ndarray    # Dynamic weights per ensemble
    regime: np.ndarray              # Detected regime per sample
    model_attribution: Dict[str, float]  # Which models contributed
```

---

## Implementation Tasks

### Task 8.1: Regime-Aware Meta-Learner
**File:** `src/models/meta_learners/regime_aware.py`

**Status:** ❌ Not implemented

**Implementation:**
```python
class RegimeAwareMetaLearner(BaseModel):
    def __init__(
        self,
        ensemble_models: List[BaseModel],
        regime_features: List[str]
    ):
        """
        Args:
            ensemble_models: Trained ensemble models (from Phase 7)
            regime_features: Features for regime detection (volatility, trend, etc.)
        """
        self.ensemble_models = ensemble_models
        self.regime_features = regime_features
        self.regime_classifier = None  # Classify regime
        self.regime_weights = {}  # Weights per regime per ensemble

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        sample_weights: Optional[np.ndarray] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> TrainingMetrics:
        """Train regime-aware meta-learner."""

        # 1. Detect regimes in training data
        regimes_train = self._detect_regimes(X_train)
        regimes_val = self._detect_regimes(X_val)

        # 2. For each regime:
        for regime in ["trending", "ranging", "volatile"]:
            # a. Filter samples for this regime
            regime_mask_train = (regimes_train == regime)
            regime_mask_val = (regimes_val == regime)

            # b. Evaluate each ensemble on this regime
            regime_performance = {}
            for ensemble in self.ensemble_models:
                preds = ensemble.predict(X_val[regime_mask_val])
                acc = np.mean(preds.predictions == y_val[regime_mask_val])
                regime_performance[ensemble.name] = acc

            # c. Optimize ensemble weights for this regime
            self.regime_weights[regime] = self._optimize_weights(
                regime_performance
            )

        # 3. Train regime classifier
        self.regime_classifier = self._train_regime_classifier(
            X_train, regimes_train
        )

        return TrainingMetrics(...)

    def predict(self, X: np.ndarray) -> MetaLearnerOutput:
        """Generate regime-aware predictions."""

        # 1. Detect regime for each sample
        regimes = self.regime_classifier.predict(X)

        # 2. Get ensemble predictions
        ensemble_preds = [
            ensemble.predict(X) for ensemble in self.ensemble_models
        ]

        # 3. For each sample, weight ensembles by regime
        final_probs = np.zeros((len(X), 3))
        ensemble_weights = np.zeros((len(X), len(self.ensemble_models)))

        for i in range(len(X)):
            regime = regimes[i]
            weights = self.regime_weights[regime]

            # Weighted average of ensemble probabilities
            for j, (ensemble_pred, weight) in enumerate(zip(ensemble_preds, weights)):
                final_probs[i] += ensemble_pred.probabilities[i] * weight
                ensemble_weights[i, j] = weight

        predictions = np.argmax(final_probs, axis=1) - 1  # Map to {-1, 0, 1}
        confidence = np.max(final_probs, axis=1)

        return MetaLearnerOutput(
            predictions=predictions,
            probabilities=final_probs,
            confidence=confidence,
            ensemble_weights=ensemble_weights,
            regime=regimes
        )

    def _detect_regimes(self, X: np.ndarray) -> np.ndarray:
        """Detect market regime for each sample."""
        # Use regime features (ADX, volatility, etc.)
        # Cluster into regimes: trending, ranging, volatile
        # Return regime labels
```

**Regime Detection Logic:**
- **Trending:** High ADX (>25), directional momentum
- **Ranging:** Low ADX (<20), mean-reverting
- **Volatile:** High ATR, large price swings

### Task 8.2: Confidence-Based Meta-Learner
**File:** `src/models/meta_learners/confidence_based.py`

**Status:** ❌ Not implemented

**Implementation:**
```python
class ConfidenceBasedMetaLearner(BaseModel):
    def __init__(
        self,
        ensemble_models: List[BaseModel],
        confidence_threshold: float = 0.6
    ):
        """
        Args:
            ensemble_models: Trained ensemble models
            confidence_threshold: Min confidence to trust prediction
        """
        self.ensemble_models = ensemble_models
        self.confidence_threshold = confidence_threshold

    def predict(self, X: np.ndarray) -> MetaLearnerOutput:
        """Weight ensembles by prediction confidence."""

        # 1. Get ensemble predictions with confidence
        ensemble_preds = [
            ensemble.predict(X) for ensemble in self.ensemble_models
        ]

        # 2. For each sample, weight by confidence
        final_probs = np.zeros((len(X), 3))
        ensemble_weights = np.zeros((len(X), len(self.ensemble_models)))

        for i in range(len(X)):
            # Collect confidences
            confidences = [pred.confidence[i] for pred in ensemble_preds]

            # Weight ensembles by confidence (softmax)
            weights = self._softmax(confidences)

            # If all confidences low, fall back to equal weighting
            if max(confidences) < self.confidence_threshold:
                weights = np.ones(len(self.ensemble_models)) / len(self.ensemble_models)

            # Weighted average
            for j, (ensemble_pred, weight) in enumerate(zip(ensemble_preds, weights)):
                final_probs[i] += ensemble_pred.probabilities[i] * weight
                ensemble_weights[i, j] = weight

        predictions = np.argmax(final_probs, axis=1) - 1
        confidence = np.max(final_probs, axis=1)

        return MetaLearnerOutput(
            predictions=predictions,
            probabilities=final_probs,
            confidence=confidence,
            ensemble_weights=ensemble_weights
        )
```

### Task 8.3: Adaptive Performance-Based Meta-Learner
**File:** `src/models/meta_learners/adaptive.py`

**Status:** ❌ Not implemented

**Implementation:**
```python
class AdaptiveMetaLearner(BaseModel):
    def __init__(
        self,
        ensemble_models: List[BaseModel],
        lookback_window: int = 500  # Bars to track performance
    ):
        """
        Args:
            ensemble_models: Trained ensemble models
            lookback_window: Recent bars to calculate performance
        """
        self.ensemble_models = ensemble_models
        self.lookback_window = lookback_window
        self.performance_history = {
            ensemble.name: deque(maxlen=lookback_window)
            for ensemble in ensemble_models
        }

    def update_performance(
        self,
        ensemble_name: str,
        prediction: int,
        actual: int
    ) -> None:
        """Update performance history after each prediction."""
        correct = (prediction == actual)
        self.performance_history[ensemble_name].append(float(correct))

    def predict(self, X: np.ndarray) -> MetaLearnerOutput:
        """Weight ensembles by recent performance."""

        # 1. Calculate recent accuracy for each ensemble
        recent_accuracies = {}
        for ensemble in self.ensemble_models:
            if len(self.performance_history[ensemble.name]) > 0:
                recent_acc = np.mean(self.performance_history[ensemble.name])
            else:
                recent_acc = 0.5  # Neutral if no history

            recent_accuracies[ensemble.name] = recent_acc

        # 2. Softmax weights from accuracies
        weights = self._softmax(list(recent_accuracies.values()))

        # 3. Get ensemble predictions
        ensemble_preds = [
            ensemble.predict(X) for ensemble in self.ensemble_models
        ]

        # 4. Weighted average
        final_probs = np.zeros((len(X), 3))
        for ensemble_pred, weight in zip(ensemble_preds, weights):
            final_probs += ensemble_pred.probabilities * weight

        predictions = np.argmax(final_probs, axis=1) - 1
        confidence = np.max(final_probs, axis=1)

        return MetaLearnerOutput(
            predictions=predictions,
            probabilities=final_probs,
            confidence=confidence,
            ensemble_weights=np.tile(weights, (len(X), 1))
        )
```

**Key Feature:** Weights adapt based on rolling window of recent performance.

### Task 8.4: Hierarchical Meta-Learner
**File:** `src/models/meta_learners/hierarchical.py`

**Status:** ❌ Not implemented

**Implementation:**
```python
class HierarchicalMetaLearner(BaseModel):
    def __init__(
        self,
        base_models: List[BaseModel],
        ensemble_models: List[BaseModel],
        meta_strategy: str = "regime_aware"
    ):
        """
        Args:
            base_models: Individual models (Phase 6)
            ensemble_models: Ensemble models (Phase 7)
            meta_strategy: How to combine ('regime_aware', 'confidence', 'adaptive')
        """
        self.base_models = base_models
        self.ensemble_models = ensemble_models
        self.meta_strategy = meta_strategy

        # Build strategy
        if meta_strategy == "regime_aware":
            self.strategy = RegimeAwareMetaLearner(ensemble_models)
        elif meta_strategy == "confidence":
            self.strategy = ConfidenceBasedMetaLearner(ensemble_models)
        elif meta_strategy == "adaptive":
            self.strategy = AdaptiveMetaLearner(ensemble_models)

    def predict(self, X: np.ndarray) -> MetaLearnerOutput:
        """Two-level prediction with fallback."""

        # 1. Get ensemble predictions
        ensemble_output = self.strategy.predict(X)

        # 2. If confidence too low, fall back to best single model
        low_confidence_mask = ensemble_output.confidence < 0.5

        if np.any(low_confidence_mask):
            # Use best base model for low-confidence samples
            best_base = self._get_best_base_model()
            fallback_preds = best_base.predict(X[low_confidence_mask])

            ensemble_output.predictions[low_confidence_mask] = fallback_preds.predictions
            ensemble_output.probabilities[low_confidence_mask] = fallback_preds.probabilities

        return ensemble_output
```

---

## Testing Requirements

### Unit Tests
**File:** `tests/models/test_meta_learners.py`

```python
def test_regime_detection():
    """Test regime detection logic."""
    # 1. Create OHLCV with known regimes (trending, ranging)
    # 2. Detect regimes
    # 3. Assert correct regime labels

def test_regime_aware_weighting():
    """Test ensemble weights vary by regime."""
    # 1. Train regime-aware meta-learner
    # 2. Predict on trending data
    # 3. Predict on ranging data
    # 4. Assert different ensemble weights

def test_confidence_based_weighting():
    """Test ensembles weighted by confidence."""
    # 1. Create ensembles with varying confidence
    # 2. Predict with confidence-based meta-learner
    # 3. Assert high-confidence ensembles weighted more

def test_adaptive_performance_tracking():
    """Test performance history updates."""
    # 1. Create adaptive meta-learner
    # 2. Update with correct/incorrect predictions
    # 3. Assert performance history updated
    # 4. Assert weights adapt to performance
```

---

## Artifacts

### Meta-Learner Models
**Location:** `experiments/runs/{run_id}/models/meta_learners/`

**Files:**
- `regime_aware_meta.pkl`
- `confidence_based_meta.pkl`
- `adaptive_meta.pkl`
- `hierarchical_meta.pkl`

### Performance Reports
```json
// regime_aware_meta_report.json
{
  "meta_learner_type": "regime_aware",
  "ensemble_models": ["voting_boosting", "stacking_boosting", "voting_neural", "stacking_neural"],
  "regime_weights": {
    "trending": {
      "voting_boosting": 0.15,
      "stacking_boosting": 0.35,
      "voting_neural": 0.10,
      "stacking_neural": 0.40
    },
    "ranging": {
      "voting_boosting": 0.30,
      "stacking_boosting": 0.25,
      "voting_neural": 0.20,
      "stacking_neural": 0.25
    },
    "volatile": {
      "voting_boosting": 0.40,
      "stacking_boosting": 0.30,
      "voting_neural": 0.15,
      "stacking_neural": 0.15
    }
  },
  "performance": {
    "overall_accuracy": 0.70,
    "regime_breakdown": {
      "trending": 0.73,
      "ranging": 0.65,
      "volatile": 0.68
    },
    "improvement_over_best_ensemble": 0.02
  }
}
```

---

## Configuration

**File:** `config/meta_learners.yaml`

```yaml
regime_aware:
  regime_features: ["adx_14", "atr_14", "bb_width", "volatility"]
  regimes: ["trending", "ranging", "volatile"]
  regime_classifier: "kmeans"  # or "random_forest", "gmm"
  n_clusters: 3

confidence_based:
  confidence_threshold: 0.6
  weighting_method: "softmax"  # or "linear", "rank"

adaptive:
  lookback_window: 500
  min_samples: 100  # Min samples before adapting
  smoothing: 0.9    # Exponential smoothing factor

hierarchical:
  meta_strategy: "regime_aware"  # or "confidence", "adaptive"
  fallback_threshold: 0.5
  fallback_model: "best_ensemble"  # or specific model name
```

---

## Expected Performance Improvements

| Meta-Learner | vs Best Ensemble | vs Best Single Model | Use Case |
|--------------|------------------|---------------------|----------|
| Regime-Aware | +1-3% | +5-8% | Changing market conditions |
| Confidence-Based | +0.5-2% | +4-7% | Uncertain predictions common |
| Adaptive | +1-2% | +4-6% | Concept drift over time |
| Hierarchical | +2-4% | +6-10% | Robust across all conditions |

**Note:** Improvements vary by dataset, symbol, and horizon.

---

## Command-Line Interface

**Script:** `scripts/train_meta_learner.py`

**Usage:**
```bash
# Train regime-aware meta-learner
python scripts/train_meta_learner.py \
  --type regime_aware \
  --ensembles voting_boosting,stacking_boosting,voting_neural,stacking_neural \
  --horizon 20

# Train confidence-based meta-learner
python scripts/train_meta_learner.py \
  --type confidence_based \
  --ensembles all \
  --confidence-threshold 0.6

# Train adaptive meta-learner with tracking
python scripts/train_meta_learner.py \
  --type adaptive \
  --ensembles all \
  --lookback-window 500
```

---

## Dependencies

**Internal:**
- Phase 7 (ensemble models)
- `src/phase1/stages/regime/` (regime detection)

**External:**
- `scikit-learn >= 1.2.0` (clustering for regimes)
- `numpy >= 1.24.0`

---

## Next Steps

**After Phase 8 completion:**
1. ✅ Meta-learners adaptively combine ensembles
2. ➡️ Deploy for live inference (Phase 9 - future)
3. ➡️ Continuous performance monitoring and adaptation

**Validation Checklist:**
- [ ] Regime detection works correctly
- [ ] Ensemble weights vary by context (regime, confidence, performance)
- [ ] Meta-learner performance exceeds best ensemble
- [ ] Adaptive weighting updates correctly
- [ ] Meta-learner saved and loadable
- [ ] Performance reports show regime/confidence breakdowns

---

## Performance

**Estimated Benchmarks (4 ensembles):**
- Regime-aware training: ~5 minutes (regime clustering + weight optimization)
- Confidence-based: Instant (no training, just weighting logic)
- Adaptive: Minimal overhead (<1 second per prediction, performance tracking)
- Hierarchical: ~5 minutes (strategy training + fallback logic)

**Memory:** +100-200 MB (stores ensemble predictions + performance history)

---

## Future Enhancements

### Phase 8.1: Online Learning
- Update meta-learner weights in real-time
- Incremental regime updates
- Continual learning from new data

### Phase 8.2: Multi-Horizon Meta-Learner
- Train meta-learner across multiple horizons (5, 10, 15, 20)
- Learn horizon-specific weighting strategies

### Phase 8.3: Contextual Bandits
- Use multi-armed bandit algorithms to select ensembles
- Explore/exploit trade-off for ensemble selection

---

## References

**Code Files (Planned):**
- `src/models/meta_learners/regime_aware.py`
- `src/models/meta_learners/confidence_based.py`
- `src/models/meta_learners/adaptive.py`
- `src/models/meta_learners/hierarchical.py`

**Config Files:**
- `config/meta_learners.yaml`

**Documentation:**
- `docs/phases/PHASE_7_ENSEMBLES.md` - Ensemble foundation

**Tests (Planned):**
- `tests/models/test_meta_learners.py`

---

## Implementation Priority

**High Priority:**
1. Regime-aware meta-learner (most impact)
2. Adaptive performance-based (handles concept drift)

**Medium Priority:**
3. Confidence-based (simple, fast)

**Low Priority:**
4. Hierarchical (complex, marginal improvement)

**Estimated Total Effort:** 5-7 days
- Regime-aware: 3 days
- Adaptive: 2 days
- Confidence-based: 1 day
- Hierarchical: 1 day
