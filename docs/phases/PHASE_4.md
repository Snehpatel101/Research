# Phase 4: Ensemble Meta-Learner (Stacking)

## Current Status: PLANNED (Not Implemented)

**IMPLEMENTATION STATUS:**
- ❌ Meta-learner implementations (Logistic, XGBoost) - Not implemented
- ❌ Stacking feature engineering - Not implemented
- ❌ Ensemble trainer - Not implemented
- ❌ Model diversity analysis - Not implemented
- ❌ Scripts (`scripts/train_ensemble.py`) - Do not exist
- ❌ Config files (`config/meta_learners/*.yaml`) - Do not exist

**DEPENDENCIES:**
- ✅ Phase 1 (Data Pipeline) - **COMPLETE**
- ❌ Phase 2 (Model Factory) - **NOT STARTED** - Required
- ❌ Phase 3 (Cross-Validation) - **NOT STARTED** - Required for OOF predictions

**BLOCKED BY:**
- Phase 4 requires OOF predictions from Phase 3
- Phase 3 requires trained models from Phase 2

**NEXT STEPS (After Phase 2 & 3 Complete):**
1. Implement stacking feature engineering
2. Implement LogisticMetaLearner and XGBoostMetaLearner
3. Create ensemble trainer script: `scripts/train_ensemble.py`
4. Implement diversity analysis tools
5. Compare ensemble vs. baseline averaging

Phase 4 trains a meta-learner that combines base model predictions into a final ensemble. The goal is to beat the best single model by intelligently weighting predictions based on model confidence, agreement, and market regime. This document covers stacking architecture, base model diversity requirements, meta-learner options, and out-of-fold prediction handling.

---

## Overview

```
OOS Predictions (Phase 3)  -->  Stacking Features  -->  Meta-Learner  -->  Ensemble Predictions
        |                             |                      |
[model1_probs, model2_probs, ...]     |        [Logistic, XGBoost, or Neural]
        |                             |
        +-----> Agreement, Confidence, Regime Features
```

**Key Principles:**
1. **Use OOS predictions only** - Never train meta-learner on in-sample base model predictions
2. **Maximize base model diversity** - Uncorrelated base models improve ensemble
3. **Keep meta-learner simple** - Avoid overfitting with complex meta-learners
4. **Preserve interpretability** - Understand why ensemble makes decisions

---

## Why Stacking Works

### Theoretical Foundation

Stacking exploits **prediction diversity** among base models:

```
Ensemble Error = Average Base Error - Diversity
```

If base models make **different errors**, the ensemble can correct them. If all models make the **same errors**, stacking cannot help.

### Practical Intuition

Each base model has different strengths:

| Model | Strength | Weakness |
|-------|----------|----------|
| XGBoost | Feature interactions, regime detection | Sequential patterns |
| LSTM | Temporal dependencies, trend detection | Noisy short-term signals |
| LightGBM | Fast training, regularization | Similar to XGBoost |
| RandomForest | Robustness, no tuning needed | Less accurate than boosting |

The meta-learner learns: "When should I trust XGBoost vs LSTM?"

### When Stacking Works vs Fails

**Stacking works when:**
- Base models are diverse (correlation < 0.7)
- Each model contributes unique signal
- Meta-learner has sufficient training data
- Base model probabilities are well-calibrated

**Stacking fails when:**
- Base models are too similar (correlation > 0.85)
- One model dominates all others
- Insufficient OOS predictions for meta-training
- Base models are poorly calibrated

---

## Base Model Diversity Requirements

### Measuring Diversity

```python
def compute_model_diversity(
    stacking_df: pd.DataFrame,
    model_names: List[str]
) -> Dict[str, float]:
    """
    Compute diversity metrics between base models.

    Returns:
        Dictionary with diversity metrics
    """
    pred_cols = [f"{model}_pred" for model in model_names]
    pred_matrix = stacking_df[pred_cols].values

    # Pairwise disagreement rate
    n_models = len(model_names)
    disagreement_matrix = np.zeros((n_models, n_models))

    for i in range(n_models):
        for j in range(i + 1, n_models):
            disagreement = (pred_matrix[:, i] != pred_matrix[:, j]).mean()
            disagreement_matrix[i, j] = disagreement
            disagreement_matrix[j, i] = disagreement

    # Average pairwise disagreement
    avg_disagreement = disagreement_matrix[np.triu_indices(n_models, k=1)].mean()

    # Prediction correlation
    corr_matrix = stacking_df[pred_cols].corr()
    avg_correlation = corr_matrix.values[np.triu_indices(n_models, k=1)].mean()

    # Double-fault diversity (both wrong simultaneously)
    y_true = stacking_df["y_true"].values
    double_fault_rates = []

    for i in range(n_models):
        for j in range(i + 1, n_models):
            both_wrong = ((pred_matrix[:, i] != y_true) & (pred_matrix[:, j] != y_true)).mean()
            double_fault_rates.append(both_wrong)

    avg_double_fault = np.mean(double_fault_rates)

    return {
        "avg_disagreement": avg_disagreement,
        "avg_correlation": avg_correlation,
        "avg_double_fault": avg_double_fault,
        "diversity_score": avg_disagreement * (1 - avg_double_fault),
        "diversity_grade": grade_diversity_score(avg_disagreement, avg_correlation)
    }


def grade_diversity_score(disagreement: float, correlation: float) -> str:
    """Grade overall diversity."""
    if disagreement > 0.3 and correlation < 0.5:
        return "Excellent"
    elif disagreement > 0.2 and correlation < 0.65:
        return "Good"
    elif disagreement > 0.15 and correlation < 0.75:
        return "Moderate"
    else:
        return "Poor (consider different base models)"
```

### Recommended Ensemble Combinations

| Ensemble Type | Models | Why It Works | Recommended Weights |
|---------------|--------|--------------|---------------------|
| **Boosting-Only** | XGBoost + LightGBM + CatBoost | Fast, interpretable, good for tabular features | 0.4 / 0.3 / 0.3 |
| **Neural-Only (RNN)** | LSTM + TCN + GRU | Temporal patterns, sequence diversity | 0.4 / 0.35 / 0.25 |
| **Transformer-Only** | PatchTST + iTransformer + TFT | Long-range dependencies, attention diversity | 0.35 / 0.35 / 0.30 |
| **Hybrid (Recommended)** | XGBoost + TCN + Transformer | Maximum diversity across paradigms | Equal or learned |
| **Full Stack** | XGB + LGB + LSTM + TCN + Transformer | 5 diverse models for robust ensemble | Learned via stacking |

### Why Each Ensemble Works

**Boosting-Only Ensemble:**
- All models excel at feature interactions and regime detection
- Fast training and inference (< 1ms per prediction)
- Highly interpretable feature importance
- Risk: Similar failure modes on temporal patterns

**Neural-Only (RNN) Ensemble:**
- LSTM: Long-term memory, trend detection
- TCN: Parallel processing, fixed receptive field, faster than RNN
- GRU: Lighter than LSTM, good for shorter sequences
- Risk: All sensitive to sequence length, need careful normalization

**Transformer-Only Ensemble:**
- PatchTST: Patches reduce compute, channel independence
- iTransformer: Inverted attention over variables, good for multivariate
- TFT: Interpretable attention, static/dynamic features
- Risk: Expensive training, need large datasets

**Hybrid Ensemble (Recommended for Production):**
- XGBoost: Fast, handles feature interactions, robust baseline
- TCN: Temporal patterns without recurrence overhead
- Transformer (PatchTST or TFT): Long-range dependencies
- Maximum diversity = Maximum error correction potential

### Stacking Architecture

```
Level 0 (Base Models):
  XGBoost     -> OOF predictions (P_xgb)     [3 class probabilities]
  LightGBM    -> OOF predictions (P_lgb)     [3 class probabilities]
  TCN         -> OOF predictions (P_tcn)     [3 class probabilities]
  Transformer -> OOF predictions (P_trans)   [3 class probabilities]

Level 1 (Meta-Learner):
  Input Features: [
      P_xgb (3), P_lgb (3), P_tcn (3), P_trans (3),    # 12 probability features
      confidence_xgb, confidence_lgb, confidence_tcn, confidence_trans,  # 4 confidence scores
      model_agreement, avg_confidence, confidence_spread  # 3 agreement features
  ]
  Total: 19 meta-features

  Meta-Learner Options:
    - LogisticRegression (recommended for simplicity)
    - XGBoost (if non-linear model interactions matter)

  Output: Final ensemble prediction [P(short), P(neutral), P(long)]
```

### Blending vs Stacking

| Approach | Weights | Pros | Cons | When to Use |
|----------|---------|------|------|-------------|
| **Blending** | Fixed (e.g., 0.4/0.3/0.3) | Simple, no overfitting risk, fast | Cannot adapt to model strengths | Few models, limited data |
| **Stacking** | Learned via meta-learner | Adapts to model expertise per regime | Risk of meta-learner overfitting | Many diverse models, large OOF dataset |
| **Weighted Average** | Fixed equal weights | Most robust, zero overfitting | Ignores model quality differences | Highly diverse models |

**Decision Guide:**
```python
def choose_ensemble_method(n_models: int, n_oof_samples: int, diversity_score: float) -> str:
    """
    Choose between blending and stacking.

    Args:
        n_models: Number of base models
        n_oof_samples: Number of OOF samples for meta-training
        diversity_score: Average model diversity (0-1)
    """
    if n_models <= 3 and diversity_score > 0.7:
        return "weighted_average"  # Few diverse models, equal weights work well

    if n_oof_samples < 10000:
        return "blending"  # Not enough data to train meta-learner safely

    if n_oof_samples >= 30000 and n_models >= 4:
        return "stacking"  # Enough data and models to benefit from learned weights

    return "blending"  # Default to simpler approach
```

### Diversity Metrics

Measure ensemble diversity before training to predict ensemble success:

```python
from dataclasses import dataclass
from typing import Dict, List
import numpy as np


@dataclass
class DiversityReport:
    """Complete diversity analysis of base models."""
    disagreement_rate: float      # How often models disagree
    prediction_correlation: float  # Correlation of predictions
    double_fault_rate: float       # Both wrong simultaneously
    diversity_score: float         # Composite score
    diversity_grade: str           # Excellent/Good/Moderate/Poor
    recommendations: List[str]


def compute_diversity_metrics(
    oof_predictions: Dict[str, np.ndarray],
    y_true: np.ndarray
) -> DiversityReport:
    """
    Compute comprehensive diversity metrics.

    Args:
        oof_predictions: {model_name: predictions array}
        y_true: True labels

    Returns:
        DiversityReport with metrics and recommendations
    """
    model_names = list(oof_predictions.keys())
    n_models = len(model_names)
    preds_matrix = np.column_stack([oof_predictions[m] for m in model_names])

    # 1. Disagreement Rate
    disagreements = []
    for i in range(n_models):
        for j in range(i + 1, n_models):
            disagree = (preds_matrix[:, i] != preds_matrix[:, j]).mean()
            disagreements.append(disagree)
    avg_disagreement = np.mean(disagreements)

    # 2. Prediction Correlation
    corr_matrix = np.corrcoef(preds_matrix.T)
    upper_tri = corr_matrix[np.triu_indices(n_models, k=1)]
    avg_correlation = np.mean(upper_tri)

    # 3. Double Fault Rate (both models wrong simultaneously)
    double_faults = []
    for i in range(n_models):
        for j in range(i + 1, n_models):
            both_wrong = ((preds_matrix[:, i] != y_true) &
                          (preds_matrix[:, j] != y_true)).mean()
            double_faults.append(both_wrong)
    avg_double_fault = np.mean(double_faults)

    # 4. Composite Diversity Score
    # High disagreement + low correlation + low double-fault = good
    diversity_score = (
        avg_disagreement * 0.4 +
        (1 - avg_correlation) * 0.3 +
        (1 - avg_double_fault) * 0.3
    )

    # 5. Grade and Recommendations
    if diversity_score > 0.5 and avg_correlation < 0.5:
        grade = "Excellent"
        recommendations = ["Stacking likely to improve over best single model"]
    elif diversity_score > 0.35 and avg_correlation < 0.65:
        grade = "Good"
        recommendations = ["Ensemble should provide modest improvement"]
    elif diversity_score > 0.25:
        grade = "Moderate"
        recommendations = [
            "Consider replacing similar models",
            "Add models from different families (e.g., tree + neural)"
        ]
    else:
        grade = "Poor"
        recommendations = [
            "Models too similar for effective ensemble",
            "Replace boosting models with RNN/Transformer",
            "Use different feature subsets per model"
        ]

    return DiversityReport(
        disagreement_rate=avg_disagreement,
        prediction_correlation=avg_correlation,
        double_fault_rate=avg_double_fault,
        diversity_score=diversity_score,
        diversity_grade=grade,
        recommendations=recommendations
    )
```

### Recommended Base Model Combinations (Legacy Reference)

| Combination | Diversity | Rationale |
|-------------|-----------|-----------|
| XGBoost + LSTM + RF | Excellent | Tree + RNN + Bagging |
| XGBoost + LightGBM + CatBoost | Poor | All boosting, similar errors |
| XGBoost + LSTM + Transformer | Good | Tree + RNN + Attention |
| LightGBM + GRU + Logistic | Good | Fast + Sequential + Linear |

### Selecting Diverse Base Models

```python
def select_diverse_base_models(
    cv_results: Dict[Tuple[str, int], CVResult],
    min_performance: float = 0.35,  # Minimum F1
    max_correlation: float = 0.7,
    n_models: int = 3
) -> List[str]:
    """
    Select diverse base models for ensemble.

    Strategy:
    1. Filter models meeting minimum performance
    2. Start with best model
    3. Add models with lowest correlation to existing set
    """
    # Compute pairwise correlations
    model_names = list(set(m for m, h in cv_results.keys()))
    correlations = compute_pairwise_correlations(cv_results)

    # Filter by performance
    qualified = [m for m in model_names if get_avg_f1(cv_results, m) >= min_performance]

    if len(qualified) < n_models:
        raise ValueError(f"Only {len(qualified)} models meet minimum F1 threshold")

    # Greedy selection for diversity
    selected = [max(qualified, key=lambda m: get_avg_f1(cv_results, m))]

    while len(selected) < n_models:
        # Find model with lowest average correlation to selected
        best_candidate = None
        best_avg_corr = float('inf')

        for candidate in qualified:
            if candidate in selected:
                continue

            avg_corr = np.mean([correlations[candidate][s] for s in selected])
            if avg_corr < best_avg_corr and avg_corr < max_correlation:
                best_avg_corr = avg_corr
                best_candidate = candidate

        if best_candidate is None:
            break

        selected.append(best_candidate)

    return selected
```

---

## Stacking Feature Engineering

### Basic Stacking Features

For each base model, include class probabilities:

```python
def create_basic_stacking_features(
    stacking_df: pd.DataFrame,
    model_names: List[str]
) -> pd.DataFrame:
    """
    Create basic stacking features from OOS predictions.

    Features per model (3 models x 3 classes = 9 features):
    - prob_short, prob_neutral, prob_long
    """
    feature_cols = []

    for model in model_names:
        for cls in ["short", "neutral", "long"]:
            col = f"{model}_prob_{cls}"
            feature_cols.append(col)

    return stacking_df[feature_cols].copy()
```

### Extended Stacking Features

Add derived features that help meta-learner understand model behavior:

```python
def create_extended_stacking_features(
    stacking_df: pd.DataFrame,
    model_names: List[str],
    include_regime: bool = True
) -> pd.DataFrame:
    """
    Create extended stacking features.

    Extended features:
    - Model confidence (max probability)
    - Model entropy (prediction uncertainty)
    - Agreement features (do models agree?)
    - Regime features (volatility, trend)
    """
    df = create_basic_stacking_features(stacking_df, model_names)

    # Per-model derived features
    for model in model_names:
        prob_cols = [f"{model}_prob_short", f"{model}_prob_neutral", f"{model}_prob_long"]
        probs = stacking_df[prob_cols].values

        # Confidence (max probability)
        df[f"{model}_confidence"] = probs.max(axis=1)

        # Entropy (uncertainty)
        df[f"{model}_entropy"] = -np.sum(probs * np.log(probs + 1e-10), axis=1)

        # Predicted class
        df[f"{model}_pred"] = probs.argmax(axis=1) - 1  # -1, 0, 1

        # Margin (difference between top 2 probabilities)
        sorted_probs = np.sort(probs, axis=1)
        df[f"{model}_margin"] = sorted_probs[:, -1] - sorted_probs[:, -2]

    # Agreement features
    pred_cols = [f"{model}_pred" for model in model_names]
    preds = df[pred_cols].values

    # All models agree
    df["all_agree"] = (np.abs(preds[:, 0] - preds).max(axis=1) == 0).astype(float)

    # Majority vote
    df["majority_pred"] = np.apply_along_axis(
        lambda x: np.bincount(x.astype(int) + 1, minlength=3).argmax() - 1,
        axis=1, arr=preds
    )

    # Agreement count (how many models agree with majority)
    df["agreement_count"] = np.apply_along_axis(
        lambda x: np.bincount(x.astype(int) + 1, minlength=3).max(),
        axis=1, arr=preds
    )

    # Average confidence
    conf_cols = [f"{model}_confidence" for model in model_names]
    df["avg_confidence"] = df[conf_cols].mean(axis=1)

    # Confidence spread (disagreement in confidence)
    df["confidence_spread"] = df[conf_cols].std(axis=1)

    # Regime features (if available in stacking data)
    if include_regime and "volatility_regime" in stacking_df.columns:
        df["vol_regime"] = stacking_df["volatility_regime"]
        df["trend_regime"] = stacking_df["trend_regime"]

    return df
```

### Feature Importance for Meta-Learner

```python
def analyze_meta_feature_importance(
    meta_learner,
    feature_names: List[str]
) -> pd.DataFrame:
    """Analyze which features matter for meta-learner."""

    if hasattr(meta_learner, 'coef_'):
        # Logistic regression
        importance = np.abs(meta_learner.coef_).mean(axis=0)
    elif hasattr(meta_learner, 'feature_importances_'):
        # Tree-based
        importance = meta_learner.feature_importances_
    else:
        return pd.DataFrame()

    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importance
    }).sort_values("importance", ascending=False)

    df["cumulative"] = df["importance"].cumsum() / df["importance"].sum()

    return df
```

---

## Meta-Learner Options

### Option 1: Logistic Regression (Recommended)

**Best for:** Well-calibrated base models, interpretability needs

```python
from sklearn.linear_model import LogisticRegression


class LogisticMetaLearner:
    """
    Logistic regression meta-learner.

    Pros:
    - Simple, fast, interpretable
    - L2 regularization prevents overfitting
    - Coefficients show model weights

    Cons:
    - Cannot capture non-linear interactions
    - Assumes linear relationship between base probs and outcome
    """

    def __init__(self, C: float = 1.0):
        self.model = LogisticRegression(
            multi_class='multinomial',
            solver='lbfgs',
            C=C,
            max_iter=500,
            random_state=42
        )
        self.feature_names = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.feature_names = X.columns.tolist()
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def get_model_weights(self, model_names: List[str]) -> pd.DataFrame:
        """Extract per-model weights from coefficients."""
        weights = []

        for model in model_names:
            model_cols = [c for c in self.feature_names if c.startswith(f"{model}_prob_")]
            col_indices = [self.feature_names.index(c) for c in model_cols]

            # Average absolute coefficient across classes
            model_weight = np.abs(self.model.coef_[:, col_indices]).mean()
            weights.append({"model": model, "weight": model_weight})

        df = pd.DataFrame(weights)
        df["weight_normalized"] = df["weight"] / df["weight"].sum()
        return df.sort_values("weight", ascending=False)
```

**Configuration:**
```yaml
# config/meta_learners/logistic.yaml
meta_learner: logistic
C: 1.0  # Regularization strength (lower = more regularization)
max_iter: 500
feature_set: "basic"  # or "extended"
```

### Option 2: XGBoost Meta-Learner

**Best for:** Non-linear model interactions, regime-adaptive weighting

```python
from xgboost import XGBClassifier


class XGBoostMetaLearner:
    """
    XGBoost meta-learner.

    Pros:
    - Captures non-linear interactions
    - Can use regime features effectively
    - Feature importance built-in

    Cons:
    - Risk of overfitting (use shallow trees)
    - Less interpretable than logistic
    """

    def __init__(
        self,
        max_depth: int = 3,
        n_estimators: int = 100,
        learning_rate: float = 0.1
    ):
        self.model = XGBClassifier(
            max_depth=max_depth,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            objective='multi:softprob',
            num_class=3,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            use_label_encoder=False,
            eval_metric='mlogloss'
        )
        self.feature_names = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None
    ):
        self.feature_names = X.columns.tolist()

        eval_set = None
        if X_val is not None:
            eval_set = [(X_val, y_val)]

        self.model.fit(
            X, y,
            eval_set=eval_set,
            verbose=False
        )
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def get_feature_importance(self) -> pd.DataFrame:
        importance = self.model.feature_importances_
        df = pd.DataFrame({
            "feature": self.feature_names,
            "importance": importance
        }).sort_values("importance", ascending=False)
        return df
```

**Configuration:**
```yaml
# config/meta_learners/xgboost.yaml
meta_learner: xgboost
max_depth: 3  # Keep shallow to prevent overfitting
n_estimators: 100
learning_rate: 0.1
subsample: 0.8
colsample_bytree: 0.8
feature_set: "extended"  # XGBoost can handle more features
```

### Option 3: Neural Network Meta-Learner

**Best for:** Complex patterns, large stacking datasets

```python
import torch
import torch.nn as nn


class NeuralMetaLearner(nn.Module):
    """
    Simple neural network meta-learner.

    Pros:
    - Can learn complex patterns
    - Flexible architecture

    Cons:
    - Needs more data
    - Harder to interpret
    - More hyperparameters
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 32,
        dropout: float = 0.3,
        num_classes: int = 3
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
```

### Baseline: Simple Averaging

Always compare against simple averaging:

```python
class AveragingBaseline:
    """Simple averaging baseline - stacking must beat this."""

    def __init__(self, model_names: List[str]):
        self.model_names = model_names

    def predict_proba(self, stacking_df: pd.DataFrame) -> np.ndarray:
        """Average probabilities across all models."""
        probs = np.zeros((len(stacking_df), 3))

        for model in self.model_names:
            prob_cols = [f"{model}_prob_short", f"{model}_prob_neutral", f"{model}_prob_long"]
            probs += stacking_df[prob_cols].values

        probs /= len(self.model_names)
        return probs

    def predict(self, stacking_df: pd.DataFrame) -> np.ndarray:
        probs = self.predict_proba(stacking_df)
        return probs.argmax(axis=1) - 1  # -1, 0, 1
```

---

## Training Process

### Complete Training Pipeline

```python
from dataclasses import dataclass
from typing import Dict, Optional
import pandas as pd
import numpy as np


@dataclass
class EnsembleConfig:
    """Configuration for ensemble training."""
    meta_learner_type: str = "logistic"  # logistic, xgboost, neural
    feature_set: str = "extended"  # basic, extended
    validation_split: float = 0.2  # Meta-train/meta-val split
    horizons: List[int] = None

    def __post_init__(self):
        if self.horizons is None:
            self.horizons = [5, 10, 15, 20]


class EnsembleTrainer:
    """
    Trains ensemble meta-learners from stacking dataset.
    """

    def __init__(self, config: EnsembleConfig):
        self.config = config
        self.meta_learners = {}  # Per-horizon meta-learners
        self.baselines = {}  # Per-horizon baselines

    def train(
        self,
        stacking_datasets: Dict[int, pd.DataFrame],
        model_names: List[str]
    ) -> Dict[str, any]:
        """
        Train meta-learners for all horizons.

        Args:
            stacking_datasets: OOS predictions from Phase 3 (per horizon)
            model_names: Names of base models

        Returns:
            Training results and metrics
        """
        results = {}

        for horizon in self.config.horizons:
            print(f"Training meta-learner for H{horizon}...")

            stacking_df = stacking_datasets[horizon]
            result = self._train_single_horizon(stacking_df, model_names, horizon)
            results[horizon] = result

            self.meta_learners[horizon] = result["meta_learner"]
            self.baselines[horizon] = result["baseline"]

        return results

    def _train_single_horizon(
        self,
        stacking_df: pd.DataFrame,
        model_names: List[str],
        horizon: int
    ) -> Dict:
        """Train meta-learner for single horizon."""

        # Create stacking features
        if self.config.feature_set == "basic":
            X = create_basic_stacking_features(stacking_df, model_names)
        else:
            X = create_extended_stacking_features(stacking_df, model_names)

        y = stacking_df["y_true"]

        # Split into meta-train and meta-val
        split_idx = int(len(X) * (1 - self.config.validation_split))
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

        # Train meta-learner
        if self.config.meta_learner_type == "logistic":
            meta_learner = LogisticMetaLearner(C=1.0)
        elif self.config.meta_learner_type == "xgboost":
            meta_learner = XGBoostMetaLearner(max_depth=3)
        else:
            raise ValueError(f"Unknown meta-learner: {self.config.meta_learner_type}")

        meta_learner.fit(X_train, y_train)

        # Train baseline
        baseline = AveragingBaseline(model_names)

        # Evaluate on meta-val
        metrics = self._evaluate(
            meta_learner, baseline, X_val, y_val,
            stacking_df.iloc[split_idx:]
        )

        return {
            "meta_learner": meta_learner,
            "baseline": baseline,
            "metrics": metrics,
            "feature_names": X.columns.tolist(),
            "train_size": len(X_train),
            "val_size": len(X_val)
        }

    def _evaluate(
        self,
        meta_learner,
        baseline,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        stacking_val: pd.DataFrame
    ) -> Dict:
        """Evaluate meta-learner against baseline."""

        # Meta-learner predictions
        meta_probs = meta_learner.predict_proba(X_val)
        meta_preds = meta_probs.argmax(axis=1) - 1

        # Baseline predictions
        baseline_probs = baseline.predict_proba(stacking_val)
        baseline_preds = baseline_probs.argmax(axis=1) - 1

        y_true = y_val.values

        # Classification metrics
        from sklearn.metrics import f1_score, accuracy_score

        meta_f1 = f1_score(y_true, meta_preds, average='macro')
        meta_acc = accuracy_score(y_true, meta_preds)

        baseline_f1 = f1_score(y_true, baseline_preds, average='macro')
        baseline_acc = accuracy_score(y_true, baseline_preds)

        # Trading metrics (simplified)
        meta_sharpe = compute_sharpe(y_true, meta_preds, meta_probs)
        baseline_sharpe = compute_sharpe(y_true, baseline_preds, baseline_probs)

        return {
            "meta_learner": {
                "f1": meta_f1,
                "accuracy": meta_acc,
                "sharpe": meta_sharpe
            },
            "baseline": {
                "f1": baseline_f1,
                "accuracy": baseline_acc,
                "sharpe": baseline_sharpe
            },
            "improvement": {
                "f1_gain": meta_f1 - baseline_f1,
                "sharpe_gain": meta_sharpe - baseline_sharpe,
                "beats_baseline": meta_sharpe > baseline_sharpe
            }
        }

    def predict(
        self,
        stacking_df: pd.DataFrame,
        horizon: int,
        model_names: List[str]
    ) -> np.ndarray:
        """Generate ensemble predictions."""

        if self.config.feature_set == "basic":
            X = create_basic_stacking_features(stacking_df, model_names)
        else:
            X = create_extended_stacking_features(stacking_df, model_names)

        return self.meta_learners[horizon].predict_proba(X)
```

---

## Handling Overfitting

### Signs of Meta-Learner Overfitting

1. **Large gap between meta-train and meta-val performance**
2. **Meta-learner worse than simple averaging**
3. **Unstable predictions across similar inputs**

### Prevention Strategies

```python
def prevent_overfitting(
    meta_learner_type: str,
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> Dict:
    """
    Select hyperparameters to prevent overfitting.

    Rule of thumb: Keep meta-learner simpler than base models.
    """

    n_samples = len(X_train)
    n_features = X_train.shape[1]

    if meta_learner_type == "logistic":
        # Increase regularization for small datasets
        if n_samples < 5000:
            C = 0.1  # Strong regularization
        elif n_samples < 20000:
            C = 0.5
        else:
            C = 1.0
        return {"C": C}

    elif meta_learner_type == "xgboost":
        # Reduce complexity for small datasets
        if n_samples < 5000:
            return {"max_depth": 2, "n_estimators": 50}
        elif n_samples < 20000:
            return {"max_depth": 3, "n_estimators": 100}
        else:
            return {"max_depth": 4, "n_estimators": 150}

    else:
        return {}


def validate_no_overfitting(
    train_metrics: Dict,
    val_metrics: Dict,
    max_gap: float = 0.1
) -> bool:
    """
    Check if meta-learner is overfitting.

    Returns True if acceptable, False if overfitting detected.
    """
    f1_gap = train_metrics["f1"] - val_metrics["f1"]
    sharpe_gap = train_metrics["sharpe"] - val_metrics["sharpe"]

    if f1_gap > max_gap:
        print(f"WARNING: F1 gap {f1_gap:.3f} exceeds threshold")
        return False

    if sharpe_gap > max_gap * 2:  # Sharpe more variable
        print(f"WARNING: Sharpe gap {sharpe_gap:.3f} exceeds threshold")
        return False

    return True
```

---

## Calibration and Confidence

### Probability Calibration

Base model probabilities may not be well-calibrated. Consider recalibration:

```python
from sklearn.calibration import CalibratedClassifierCV


def calibrate_base_model_probs(
    model,
    X_calibration: np.ndarray,
    y_calibration: np.ndarray,
    method: str = "isotonic"
) -> CalibratedClassifierCV:
    """
    Calibrate base model probabilities using isotonic regression.

    Use when base model confidence doesn't match actual accuracy.
    """
    calibrated = CalibratedClassifierCV(
        model,
        method=method,
        cv="prefit"
    )
    calibrated.fit(X_calibration, y_calibration)
    return calibrated


def check_calibration(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    n_bins: int = 10
) -> Dict:
    """
    Check probability calibration using reliability diagram data.

    Perfect calibration: predicted probability = actual frequency
    """
    from sklearn.calibration import calibration_curve

    results = {}
    for class_idx, class_name in enumerate(["short", "neutral", "long"]):
        y_binary = (y_true == (class_idx - 1)).astype(int)
        prob_true, prob_pred = calibration_curve(
            y_binary,
            y_probs[:, class_idx],
            n_bins=n_bins
        )

        # Expected Calibration Error (ECE)
        ece = np.abs(prob_true - prob_pred).mean()

        results[class_name] = {
            "prob_true": prob_true.tolist(),
            "prob_pred": prob_pred.tolist(),
            "ece": ece
        }

    results["avg_ece"] = np.mean([r["ece"] for r in results.values() if "ece" in r])
    return results
```

---

## Output Structure

### Directory Layout

```
models/ensemble/
|
+-- h5/
|   +-- meta_learner.pkl             # Trained meta-learner
|   +-- meta_config.json             # Meta-learner configuration
|   +-- feature_importance.json      # Feature importance
|   +-- model_weights.json           # Per-model weights
|   +-- training_metrics.json        # Training/validation metrics
|
+-- h10/
|   +-- (same structure)
|
+-- h15/
|   +-- (same structure)
|
+-- h20/
|   +-- (same structure)
|
+-- ensemble_summary.json            # Overall ensemble summary
+-- diversity_analysis.json          # Base model diversity metrics

predictions/ensemble/
|
+-- val_ensemble_probs_h5.npy        # Ensemble probabilities
+-- val_ensemble_preds_h5.npy        # Ensemble predictions
+-- ...

reports/phase4/
|
+-- ensemble_performance.html        # Interactive performance report
+-- model_weights.png                # Visualization of model weights
+-- calibration_curves.png           # Probability calibration
+-- ensemble_vs_baseline.png         # Performance comparison
```

### Metrics Schema

```json
{
  "horizon": 20,
  "meta_learner_type": "logistic",
  "base_models": ["xgboost", "lightgbm", "lstm"],

  "diversity_metrics": {
    "avg_disagreement": 0.28,
    "avg_correlation": 0.52,
    "diversity_grade": "Good"
  },

  "performance": {
    "meta_learner": {
      "f1": 0.47,
      "accuracy": 0.49,
      "sharpe": 0.68
    },
    "baseline_averaging": {
      "f1": 0.44,
      "accuracy": 0.46,
      "sharpe": 0.65
    },
    "best_single_model": {
      "model": "xgboost",
      "f1": 0.46,
      "sharpe": 0.62
    }
  },

  "improvement_over_best_single": {
    "sharpe_gain": 0.06,
    "f1_gain": 0.01,
    "percent_improvement": 9.7
  },

  "model_weights": {
    "xgboost": 0.42,
    "lightgbm": 0.35,
    "lstm": 0.23
  },

  "top_features": [
    {"feature": "xgboost_prob_long", "importance": 0.15},
    {"feature": "lstm_confidence", "importance": 0.12},
    {"feature": "all_agree", "importance": 0.10}
  ]
}
```

---

## When Stacking Does Not Help

### Diagnostic Checklist

```python
def diagnose_stacking_failure(
    ensemble_metrics: Dict,
    baseline_metrics: Dict,
    diversity_metrics: Dict
) -> List[str]:
    """
    Diagnose why stacking is not improving performance.

    Returns list of potential issues.
    """
    issues = []

    # Check if ensemble beats baseline
    if ensemble_metrics["sharpe"] <= baseline_metrics["sharpe"]:
        issues.append("Ensemble does not beat simple averaging")

    # Check diversity
    if diversity_metrics["avg_correlation"] > 0.75:
        issues.append(f"Base models too similar (correlation: {diversity_metrics['avg_correlation']:.2f})")

    if diversity_metrics["avg_disagreement"] < 0.15:
        issues.append(f"Low disagreement ({diversity_metrics['avg_disagreement']:.2f}) - models make same errors")

    # Check overfitting
    if ensemble_metrics.get("train_sharpe", 0) - ensemble_metrics["sharpe"] > 0.15:
        issues.append("Meta-learner is overfitting")

    # Check calibration
    if ensemble_metrics.get("avg_ece", 0) > 0.15:
        issues.append("Base model probabilities poorly calibrated")

    return issues


def recommend_fixes(issues: List[str]) -> List[str]:
    """Recommend fixes based on diagnosed issues."""
    fixes = []

    if "too similar" in str(issues).lower():
        fixes.append("Replace one base model with different family (e.g., replace LightGBM with LSTM)")

    if "same errors" in str(issues).lower():
        fixes.append("Train base models on different feature subsets")

    if "overfitting" in str(issues).lower():
        fixes.append("Use simpler meta-learner (logistic instead of XGBoost)")
        fixes.append("Increase regularization")

    if "calibration" in str(issues).lower():
        fixes.append("Apply isotonic calibration to base model probabilities")

    if not fixes:
        fixes.append("Consider using best single model instead of ensemble")

    return fixes
```

### Fallback Strategy

```python
def select_final_model(
    ensemble_metrics: Dict,
    baseline_metrics: Dict,
    single_model_metrics: Dict[str, Dict]
) -> Dict:
    """
    Select final model for production.

    Priority:
    1. Ensemble (if beats baseline and best single)
    2. Best single model (if ensemble fails)
    """
    best_single = max(single_model_metrics.items(), key=lambda x: x[1]["sharpe"])
    best_single_name, best_single_metrics = best_single

    # Ensemble must beat baseline AND best single
    if (ensemble_metrics["sharpe"] > baseline_metrics["sharpe"] and
        ensemble_metrics["sharpe"] > best_single_metrics["sharpe"]):
        return {
            "selection": "ensemble",
            "reason": f"Ensemble Sharpe {ensemble_metrics['sharpe']:.3f} beats best single {best_single_metrics['sharpe']:.3f}",
            "sharpe": ensemble_metrics["sharpe"]
        }
    else:
        return {
            "selection": best_single_name,
            "reason": f"Best single model {best_single_name} with Sharpe {best_single_metrics['sharpe']:.3f}",
            "sharpe": best_single_metrics["sharpe"]
        }
```

---

## Success Criteria

| Criterion | Target | Measurement |
|-----------|--------|-------------|
| Ensemble Sharpe | > best single model + 0.05 | Validation metrics |
| Ensemble F1 | >= best single model | Validation metrics |
| Beats baseline | Ensemble > simple averaging | Comparison |
| No overfitting | Train-val gap < 10% | Metric comparison |
| Diversity | Base model correlation < 0.7 | Correlation matrix |
| Interpretability | Clear model weights | Coefficient analysis |

---

## Expected Performance

| Method | H5 Sharpe | H10 Sharpe | H15 Sharpe | H20 Sharpe |
|--------|-----------|------------|------------|------------|
| Best Single Model | 0.45 | 0.55 | 0.62 | 0.70 |
| Simple Averaging | 0.48 | 0.58 | 0.65 | 0.73 |
| Logistic Meta | 0.50 | 0.62 | 0.68 | 0.78 |
| XGBoost Meta | 0.52 | 0.64 | 0.70 | 0.80 |

**Typical improvement: 5-15% over best single model**

---

## Usage Examples

**NOTE: These scripts do not currently exist. Phase 4 requires Phase 2 and Phase 3 to be completed first.**

```bash
# PLANNED (not yet implemented):
# python scripts/train_ensemble.py --horizons 5,10,15,20

# CURRENT STATUS:
# Phase 4 is PLANNED and cannot be run until Phase 2 and Phase 3 are complete.
#
# Implementation order:
# 1. Complete Phase 2: Train base models (XGBoost, LightGBM, LSTM, etc.)
# 2. Complete Phase 3: Generate out-of-fold predictions via cross-validation
# 3. Implement Phase 4: Train meta-learner on OOF predictions
# 4. Proceed to Phase 5 for test set evaluation

# Current implementation:
# Only Phase 1 data pipeline is implemented:
./pipeline run --symbols MES,MGC
```

---

## Next Step

Phase 4 ensemble feeds into Phase 5 (Test Set Evaluation) for final out-of-sample performance measurement on the held-out test set that has never been seen by any model.
