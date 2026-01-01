# Feature Selection Methods for Financial Machine Learning

**Research Date:** 2024-12-24
**Purpose:** Establish best practices for feature selection in the ML Model Factory
**Target:** ~150-200 features from OHLCV futures data (MES, MGC)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Feature Selection Methods Overview](#2-feature-selection-methods-overview)
3. [Time-Series Specific Considerations](#3-time-series-specific-considerations)
4. [Multi-Collinearity Handling](#4-multi-collinearity-handling)
5. [Horizon-Specific Feature Selection](#5-horizon-specific-feature-selection)
6. [Sample Size and Overfitting Guidelines](#6-sample-size-and-overfitting-guidelines)
7. [Lopez de Prado's Recommendations](#7-lopez-de-prados-recommendations)
8. [Recommended Feature Selection Pipeline](#8-recommended-feature-selection-pipeline)
9. [Implementation Code](#9-implementation-code)
10. [Sources](#10-sources)

---

## 1. Executive Summary

### Current Factory Configuration
- **Correlation threshold:** 0.70 (should increase to 0.80-0.85)
- **Variance threshold:** 0.01 (appropriate)
- **Features:** ~150-200 from technical indicators, wavelets, microstructure

### Key Recommendations

| Parameter | Current | Recommended | Rationale |
|-----------|---------|-------------|-----------|
| Correlation threshold | 0.70 | 0.80-0.85 | Industry standard; preserves more information |
| VIF threshold | N/A | 5-10 | Add as secondary multicollinearity check |
| Sample:Feature ratio | N/A | 10:1 minimum | ~50K samples supports 50-100 features |
| Selection method | Filter only | Filter + Embedded | Add LASSO or tree importance |

### Recommended Pipeline

```
Raw Features (150-200)
    |
    v
[1. Variance Filter] --> Remove near-constant (var < 0.01)
    |
    v
[2. Hierarchical Clustering] --> Group correlated features
    |
    v
[3. Clustered Feature Importance] --> MDI/MDA per cluster
    |
    v
[4. LASSO/Elastic Net] --> Embedded selection
    |
    v
[5. Walk-Forward Validation] --> Stability check
    |
    v
Selected Features (50-80)
```

---

## 2. Feature Selection Methods Overview

### 2.1 Filter Methods

Filter methods evaluate features independently of any learning algorithm.

#### Correlation-Based Filtering
- **How it works:** Remove features with correlation > threshold to reduce redundancy
- **Threshold:** 0.80-0.90 is standard; literature suggests 0.8-0.9 as the cut-off for high correlation
- **Pros:** Fast, interpretable, prevents multicollinearity
- **Cons:** Ignores feature-target relationship

#### Variance Filtering
- **How it works:** Remove features with variance below threshold
- **Current setting:** 0.01 (appropriate)
- **Note:** Normalize variance by scale for fair comparison across features

#### Mutual Information
- **Advantage:** Captures nonlinear relationships that correlation misses
- **Key insight:** "Mutual information can effectively represent dependencies of features and is one of the widely used measurements in feature selection"
- **Use case:** Particularly important for financial data where relationships are often nonlinear
- **Methods:** JMI (Joint Mutual Information) and MRMR (Minimum Redundancy Maximum Relevance) are recommended

### 2.2 Wrapper Methods

Wrapper methods use a predictive model to evaluate feature subsets iteratively.

#### Recursive Feature Elimination (RFE)
- **How it works:** Train model, rank features by importance, remove least important, repeat
- **Financial application:** Used in stock price prediction and credit scoring
- **Best practice:** Use RFECV (cross-validated RFE) with scikit-learn
- **Base models:** Random Forest or XGBoost work well as base estimators

```python
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier

rfecv = RFECV(
    estimator=RandomForestClassifier(n_estimators=100, random_state=42),
    step=1,
    cv=TimeSeriesSplit(n_splits=5),
    scoring='accuracy',
    min_features_to_select=20
)
```

### 2.3 Embedded Methods

Embedded methods perform feature selection as part of model training.

#### LASSO (L1 Regularization)
- **Mechanism:** L1 penalty shrinks coefficients to exactly zero, eliminating features
- **Financial results:** "80-85% reduction in original feature set frequently maintains or enhances model performance"
- **Use case:** Excellent for high-dimensional financial data
- **Tuning:** Use cross-validation to select optimal alpha parameter

#### Elastic Net
- **When to use:** When features are correlated (combines L1 + L2)
- **Advantage:** "Unlike pure LASSO which tends to select one variable from correlated features, Elastic Net can select multiple related variables"

#### Tree-Based Importance
- **Methods:** Feature importance from Random Forest, XGBoost, LightGBM
- **Caution:** MDI (Mean Decrease Impurity) has known biases (see Section 7)
- **Better alternative:** Permutation importance or SHAP values

---

## 3. Time-Series Specific Considerations

### 3.1 Temporal Dependency in Selection

Standard cross-validation violates the temporal order assumption in financial data.

#### The Problem
"Standard cross-validation assumes that observations are independently and identically distributed (IID), which often does not hold in time series or financial datasets."

#### Solution: Purged Cross-Validation
- **Purging:** Remove training samples whose labels overlap with test samples
- **Embargo:** Exclude samples after test fold to prevent serial correlation leakage
- **Current factory settings:**
  - Purge: 60 bars (3x max horizon)
  - Embargo: 1440 bars (~5 days)

### 3.2 Walk-Forward Feature Selection

Instead of selecting features once on all data, select features in each walk-forward window.

```python
def walk_forward_feature_selection(X, y, n_splits=5):
    """
    Perform feature selection within each walk-forward fold.
    Track feature stability across folds.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    feature_counts = defaultdict(int)

    for train_idx, test_idx in tscv.split(X):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]

        # Apply purging
        purge_idx = get_purge_indices(train_idx, test_idx, purge_window=60)
        X_train = X_train.drop(purge_idx)
        y_train = y_train.drop(purge_idx)

        # Select features on this fold
        selected = select_features_single_fold(X_train, y_train)

        for feat in selected:
            feature_counts[feat] += 1

    # Keep features selected in majority of folds
    stable_features = [f for f, count in feature_counts.items()
                       if count >= n_splits * 0.6]

    return stable_features
```

### 3.3 Feature Stability Over Time

#### Why It Matters
"Parameter instability is widely recognized as a crucial issue in forecasting. The empirical evidence of parameter instability is widespread in financial forecasting."

#### How to Measure
1. **Rolling window selection:** Apply selection on rolling 6-month windows
2. **Jaccard similarity:** Measure overlap between feature sets across windows
3. **Stability threshold:** Keep features selected in >60% of windows

```python
def compute_feature_stability(feature_sets: List[Set[str]]) -> Dict[str, float]:
    """
    Compute stability score for each feature across time windows.
    """
    all_features = set().union(*feature_sets)
    stability = {}

    for feat in all_features:
        presence_count = sum(1 for fs in feature_sets if feat in fs)
        stability[feat] = presence_count / len(feature_sets)

    return stability
```

### 3.4 Target Leakage Prevention During Selection

**Critical:** Never use future information when selecting features.

- All feature selection must use only training data
- Validation/test sets are only for final evaluation
- Purge samples with overlapping label windows

---

## 4. Multi-Collinearity Handling

### 4.1 Optimal Correlation Threshold

#### Research Findings
- **0.8-0.9:** "A bivariate correlation of 0.8 or 0.9 is commonly used as a cut-off to indicate a high correlation"
- **Current setting:** 0.70 is more aggressive than necessary
- **Recommendation:** Increase to 0.80-0.85

| Threshold | Features Removed | Information Loss | Recommendation |
|-----------|-----------------|------------------|----------------|
| 0.70 | High | Moderate | Too aggressive |
| 0.80 | Moderate | Low | Recommended |
| 0.85 | Lower | Very Low | Conservative |
| 0.90 | Minimal | Minimal | May have redundancy |

### 4.2 VIF (Variance Inflation Factor)

VIF measures how much the variance of a coefficient is inflated due to multicollinearity.

#### Thresholds
| VIF Value | Interpretation | Action |
|-----------|---------------|--------|
| 1 | No correlation | Keep |
| 1-5 | Moderate | Acceptable |
| 5-10 | High | Investigate |
| >10 | Severe | Remove |

#### Implementation
```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

def calculate_vif(X: pd.DataFrame) -> pd.DataFrame:
    """Calculate VIF for all features."""
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [
        variance_inflation_factor(X.values, i)
        for i in range(X.shape[1])
    ]
    return vif_data.sort_values("VIF", ascending=False)

def remove_high_vif_features(X: pd.DataFrame, threshold: float = 10.0) -> List[str]:
    """Iteratively remove features with highest VIF until all below threshold."""
    features_to_keep = X.columns.tolist()

    while True:
        vif = calculate_vif(X[features_to_keep])
        max_vif = vif["VIF"].max()

        if max_vif < threshold:
            break

        # Remove feature with highest VIF
        feature_to_remove = vif.loc[vif["VIF"].idxmax(), "feature"]
        features_to_keep.remove(feature_to_remove)

    return features_to_keep
```

### 4.3 Hierarchical Clustering of Features

Lopez de Prado recommends clustering correlated features and selecting one representative per cluster.

#### Approach
1. Compute Spearman correlation matrix
2. Convert to distance: `distance = 1 - |correlation|`
3. Apply hierarchical clustering (Ward's linkage)
4. Cut dendrogram at threshold to form clusters
5. Select highest-priority feature from each cluster

```python
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

def cluster_features(corr_matrix: pd.DataFrame, threshold: float = 0.5) -> Dict[int, List[str]]:
    """
    Cluster features using hierarchical clustering on correlation matrix.

    Args:
        corr_matrix: Correlation matrix of features
        threshold: Distance threshold for cutting dendrogram

    Returns:
        Dictionary mapping cluster_id to list of features
    """
    # Convert correlation to distance
    distance_matrix = 1 - np.abs(corr_matrix)
    np.fill_diagonal(distance_matrix.values, 0)

    # Hierarchical clustering
    condensed_dist = squareform(distance_matrix.values)
    linkage_matrix = linkage(condensed_dist, method='ward')

    # Cut dendrogram to form clusters
    cluster_labels = fcluster(linkage_matrix, t=threshold, criterion='distance')

    # Group features by cluster
    clusters = defaultdict(list)
    for feat, label in zip(corr_matrix.columns, cluster_labels):
        clusters[label].append(feat)

    return dict(clusters)
```

### 4.4 PCA vs. Feature Removal Tradeoffs

| Method | Pros | Cons | When to Use |
|--------|------|------|-------------|
| Feature Removal | Interpretable, simple | Loses information | When interpretability matters |
| PCA | Preserves variance | Not interpretable | Dimensionality reduction only |
| Clustering + Selection | Interpretable, groups related | Requires threshold tuning | Best for financial ML |

**Recommendation:** Use clustering + selection for this factory (preserves interpretability).

---

## 5. Horizon-Specific Feature Selection

### 5.1 Do Optimal Features Differ by Horizon?

**Yes.** Research indicates different features matter for different prediction horizons.

#### Short-Term (H5, H10 - 25-50 min)
- **Emphasis:** Momentum, microstructure, order flow
- **Key features:** RSI, MACD histogram, volume spikes, bid-ask spread proxies
- **Rationale:** "For relatively short prediction horizons of only 1 week, technical features are more important"

#### Medium-Term (H15, H20 - 75-100 min)
- **Emphasis:** Volatility regime, trend indicators, multi-timeframe
- **Key features:** ATR, ADX, MA crossovers, daily/hourly patterns
- **Rationale:** Captures regime changes and trend reversals

### 5.2 Horizon-Specific Selection Strategy

```python
HORIZON_FEATURE_WEIGHTS = {
    'h5': {
        'momentum': 1.2,    # RSI, MACD, ROC
        'microstructure': 1.3,  # spread, volume imbalance
        'volatility': 0.9,
        'trend': 0.8,
    },
    'h10': {
        'momentum': 1.1,
        'microstructure': 1.2,
        'volatility': 1.0,
        'trend': 0.9,
    },
    'h15': {
        'momentum': 1.0,
        'microstructure': 1.0,
        'volatility': 1.1,
        'trend': 1.0,
    },
    'h20': {
        'momentum': 0.9,
        'microstructure': 0.9,
        'volatility': 1.2,
        'trend': 1.1,
    },
}

def apply_horizon_weights(feature_importance: pd.Series, horizon: str) -> pd.Series:
    """
    Weight feature importance by horizon-specific preferences.
    """
    weights = HORIZON_FEATURE_WEIGHTS.get(horizon, {})

    weighted = feature_importance.copy()
    for category, weight in weights.items():
        category_features = get_features_by_category(category)
        weighted[category_features] *= weight

    return weighted
```

### 5.3 Practical Recommendation

For this factory with 4 horizons (H5, H10, H15, H20):

1. **Option A (Simple):** Use union of top features across all horizons
2. **Option B (Optimal):** Select horizon-specific feature sets

**Recommended:** Start with Option A for simplicity, then evaluate Option B if performance differs significantly across horizons.

---

## 6. Sample Size and Overfitting Guidelines

### 6.1 Sample-to-Feature Ratio Rules

| Rule | Ratio | Application |
|------|-------|-------------|
| Minimum viable | 5:1 | Basic models, low complexity |
| Standard | 10:1 | Most applications |
| Conservative | 20:1 | High-stakes, financial ML |

**For this factory (50K training samples):**
- At 10:1 ratio: **50 features maximum**
- At 20:1 ratio: **25 features maximum**
- Current ~150 features: **Significant overfitting risk**

### 6.2 The Curse of Dimensionality

"As the number of features increases, the classifier's performance improves until reaching an optimal number of features. Adding more features based on the same training set size will then degrade the classifier's performance" (Hughes Phenomenon).

#### Recommendations
1. Target 50-80 features after selection (10:1 to 6:1 ratio)
2. Use regularization (L1/L2) if keeping more features
3. Monitor train-validation gap for overfitting signals

### 6.3 Overfitting Signals

| Metric | Healthy | Warning | Critical |
|--------|---------|---------|----------|
| Train-Val Accuracy Gap | <5% | 5-10% | >10% |
| Feature Importance Stability | >0.7 | 0.5-0.7 | <0.5 |
| Cross-Validation Variance | Low | Medium | High |

---

## 7. Lopez de Prado's Recommendations

Based on "Advances in Financial Machine Learning" and "Machine Learning for Asset Managers."

### 7.1 Mean Decrease Impurity (MDI)

**Definition:** Measures the total reduction in impurity (Gini or entropy) from splits on a feature, averaged across all trees.

**Problems:**
- Biased toward features with many categories
- Uses in-sample (IS) performance
- Suffers from substitution effects (correlated features split importance)

**When to use:** Quick initial screening, but verify with MDA.

### 7.2 Mean Decrease Accuracy (MDA)

**Definition:** Measures the decrease in model accuracy when a feature is permuted (shuffled).

**Advantages:**
- Uses out-of-sample (OOS) performance
- Applicable to any classifier, not just trees
- More reliable than MDI

**Problems:**
- Also suffers from substitution effects
- Computationally more expensive

```python
from sklearn.inspection import permutation_importance

def calculate_mda(model, X_val, y_val, n_repeats=10):
    """Calculate Mean Decrease Accuracy (permutation importance)."""
    result = permutation_importance(
        model, X_val, y_val,
        n_repeats=n_repeats,
        random_state=42,
        n_jobs=-1
    )

    importance_df = pd.DataFrame({
        'feature': X_val.columns,
        'importance_mean': result.importances_mean,
        'importance_std': result.importances_std
    })

    return importance_df.sort_values('importance_mean', ascending=False)
```

### 7.3 Single Feature Importance (SFI)

**Definition:** Evaluate each feature's importance independently by training a model on that feature alone.

**Advantage:** Does not suffer from substitution effects because features are evaluated separately.

**Disadvantage:** Ignores feature interactions.

### 7.4 Clustered Feature Importance (CFI)

**The solution to substitution effects.** Lopez de Prado's recommended approach.

**Method:**
1. Cluster similar features using hierarchical clustering on correlation/mutual information
2. Apply MDI or MDA at the cluster level (shuffle/permute entire clusters)
3. Importance is attributed to the cluster, not individual features

**Results:** "The accuracy of the S&P 500 monthly returns model improves from 0.517 to 0.583 when using cMDA instead of MDA, while the AUC score improves from 0.716 to 0.779."

```python
def clustered_mda(model, X_val, y_val, feature_clusters: Dict[int, List[str]]):
    """
    Calculate Clustered MDA by permuting features at the cluster level.
    """
    baseline_score = model.score(X_val, y_val)
    cluster_importance = {}

    for cluster_id, features in feature_clusters.items():
        X_permuted = X_val.copy()

        # Permute all features in cluster together
        for feat in features:
            X_permuted[feat] = np.random.permutation(X_permuted[feat])

        permuted_score = model.score(X_permuted, y_val)
        cluster_importance[cluster_id] = baseline_score - permuted_score

    return cluster_importance
```

### 7.5 SHAP for Financial ML

**SHAP (SHapley Additive exPlanations)** provides consistent, additive feature importance.

**Advantages:**
- Theoretically grounded (game theory)
- Local and global interpretability
- Handles feature interactions

**Time Series Consideration:** Use Time-Consistent SHAP (TC-SHAP) for time series, which respects temporal dependencies.

---

## 8. Recommended Feature Selection Pipeline

### 8.1 Complete Pipeline for ML Model Factory

```
Phase 1: Data Preparation (Already Complete)
    |
    v
Phase 2: Feature Selection Pipeline
    |
    +--[Stage 1: Variance Filter]
    |      Remove features with variance < 0.01
    |      Expected: 150 -> 145 features
    |
    +--[Stage 2: Hierarchical Clustering]
    |      Cluster features with correlation > 0.80
    |      Form 20-40 clusters
    |
    +--[Stage 3: Cluster Representative Selection]
    |      Select 1 feature per cluster (highest priority)
    |      Expected: 145 -> 60-80 features
    |
    +--[Stage 4: Embedded Selection (LASSO)]
    |      Apply Elastic Net with purged CV
    |      Keep features with non-zero coefficients
    |      Expected: 60-80 -> 40-60 features
    |
    +--[Stage 5: Stability Filtering]
    |      Walk-forward selection, keep features in >60% of folds
    |      Expected: 40-60 -> 35-50 features
    |
    +--[Stage 6: Final Validation]
           Verify sample:feature ratio (50K / 40 = 1250:1 - excellent)
           Check VIF < 10 for remaining features
    |
    v
Final Selected Features (35-50)
```

### 8.2 Recommended Parameters

```python
FEATURE_SELECTION_CONFIG = {
    # Stage 1: Variance Filter
    'variance_threshold': 0.01,

    # Stage 2 & 3: Hierarchical Clustering
    'correlation_threshold': 0.80,  # Increased from 0.70
    'clustering_method': 'ward',
    'distance_metric': 'spearman',  # More robust than pearson

    # Stage 4: Embedded Selection
    'lasso_alpha_range': [0.001, 0.01, 0.1, 1.0],
    'elastic_net_l1_ratio': 0.5,  # Balance L1 and L2
    'cv_method': 'purged_kfold',
    'cv_n_splits': 5,
    'purge_window': 60,  # 3x max horizon
    'embargo_window': 1440,  # ~5 days

    # Stage 5: Stability
    'stability_threshold': 0.6,  # Feature must appear in 60% of folds
    'n_rolling_windows': 6,

    # Stage 6: Validation
    'max_vif': 10.0,
    'min_sample_feature_ratio': 10.0,

    # Horizon-specific settings
    'horizon_specific_selection': False,  # Start with unified
}
```

### 8.3 Per-Horizon Recommendations

| Horizon | Target Features | Focus Areas |
|---------|-----------------|-------------|
| H5 (25 bars) | 35-45 | Momentum, microstructure, volume |
| H10 (50 bars) | 35-45 | Momentum, volatility, trend |
| H15 (75 bars) | 40-50 | Volatility, trend, regime |
| H20 (100 bars) | 40-50 | Trend, regime, multi-timeframe |

---

## 9. Implementation Code

### 9.1 Complete Feature Selection Module

```python
"""
Enhanced Feature Selection for ML Model Factory
Based on Lopez de Prado's recommendations and 2024 research.
"""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.linear_model import LassoCV, ElasticNetCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import TimeSeriesSplit
from typing import Dict, List, Set, Tuple
from dataclasses import dataclass
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class EnhancedFeatureSelectionResult:
    """Container for enhanced feature selection results."""
    selected_features: List[str]
    removed_features: Dict[str, str]
    original_count: int
    final_count: int
    feature_clusters: Dict[int, List[str]]
    cluster_importance: Dict[int, float]
    feature_stability: Dict[str, float]
    lasso_coefficients: Dict[str, float]
    vif_scores: Dict[str, float]

    @property
    def sample_feature_ratio(self, n_samples: int = 50000) -> float:
        return n_samples / self.final_count if self.final_count > 0 else 0


class EnhancedFeatureSelector:
    """
    Multi-stage feature selection pipeline for financial ML.
    """

    def __init__(
        self,
        variance_threshold: float = 0.01,
        correlation_threshold: float = 0.80,
        lasso_cv_folds: int = 5,
        stability_threshold: float = 0.6,
        max_vif: float = 10.0,
        purge_window: int = 60,
        embargo_window: int = 1440,
    ):
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        self.lasso_cv_folds = lasso_cv_folds
        self.stability_threshold = stability_threshold
        self.max_vif = max_vif
        self.purge_window = purge_window
        self.embargo_window = embargo_window

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_priority: Dict[str, int] = None
    ) -> EnhancedFeatureSelectionResult:
        """
        Run complete feature selection pipeline.

        Args:
            X: Feature matrix
            y: Target variable
            feature_priority: Optional priority scores for tie-breaking

        Returns:
            EnhancedFeatureSelectionResult with selection details
        """
        original_features = X.columns.tolist()
        original_count = len(original_features)
        removed = {}

        logger.info(f"Starting feature selection: {original_count} features")

        # Stage 1: Variance Filter
        features_after_var = self._filter_variance(X, original_features, removed)
        logger.info(f"After variance filter: {len(features_after_var)} features")

        # Stage 2 & 3: Hierarchical Clustering + Representative Selection
        clusters = self._cluster_features(X[features_after_var])
        features_after_cluster, cluster_importance = self._select_cluster_representatives(
            X[features_after_var], y, clusters, feature_priority, removed
        )
        logger.info(f"After clustering: {len(features_after_cluster)} features")

        # Stage 4: LASSO/Elastic Net
        features_after_lasso, lasso_coefs = self._apply_lasso(
            X[features_after_cluster], y, removed
        )
        logger.info(f"After LASSO: {len(features_after_lasso)} features")

        # Stage 5: Stability Filtering
        features_after_stability, stability_scores = self._filter_by_stability(
            X[features_after_lasso], y, removed
        )
        logger.info(f"After stability filter: {len(features_after_stability)} features")

        # Stage 6: VIF Check
        final_features, vif_scores = self._check_vif(
            X[features_after_stability], removed
        )
        logger.info(f"Final features: {len(final_features)}")

        return EnhancedFeatureSelectionResult(
            selected_features=final_features,
            removed_features=removed,
            original_count=original_count,
            final_count=len(final_features),
            feature_clusters=clusters,
            cluster_importance=cluster_importance,
            feature_stability=stability_scores,
            lasso_coefficients=lasso_coefs,
            vif_scores=vif_scores,
        )

    def _filter_variance(
        self,
        X: pd.DataFrame,
        features: List[str],
        removed: Dict[str, str]
    ) -> List[str]:
        """Remove low variance features."""
        keep = []
        for feat in features:
            var = X[feat].var()
            if var >= self.variance_threshold:
                keep.append(feat)
            else:
                removed[feat] = f"low variance ({var:.4f})"
        return keep

    def _cluster_features(self, X: pd.DataFrame) -> Dict[int, List[str]]:
        """Cluster features using hierarchical clustering on correlation."""
        corr_matrix = X.corr(method='spearman')
        distance_matrix = 1 - np.abs(corr_matrix)
        np.fill_diagonal(distance_matrix.values, 0)

        condensed_dist = squareform(distance_matrix.values)
        Z = linkage(condensed_dist, method='ward')

        # Cut at threshold corresponding to 1 - correlation_threshold
        threshold = 1 - self.correlation_threshold
        labels = fcluster(Z, t=threshold, criterion='distance')

        clusters = defaultdict(list)
        for feat, label in zip(X.columns, labels):
            clusters[label].append(feat)

        return dict(clusters)

    def _select_cluster_representatives(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        clusters: Dict[int, List[str]],
        feature_priority: Dict[str, int],
        removed: Dict[str, str]
    ) -> Tuple[List[str], Dict[int, float]]:
        """Select one representative from each cluster using clustered MDA."""
        # Train a quick model for importance
        rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        rf.fit(X, y)

        # Calculate clustered importance
        cluster_importance = {}
        representatives = []

        for cluster_id, features in clusters.items():
            if len(features) == 1:
                representatives.append(features[0])
                cluster_importance[cluster_id] = rf.feature_importances_[
                    list(X.columns).index(features[0])
                ]
                continue

            # Calculate cluster importance (sum of member importances)
            cluster_imp = sum(
                rf.feature_importances_[list(X.columns).index(f)]
                for f in features
            )
            cluster_importance[cluster_id] = cluster_imp

            # Select representative by priority (or random if no priority)
            if feature_priority:
                best = max(features, key=lambda f: feature_priority.get(f, 0))
            else:
                # Use feature with highest individual importance
                best = max(
                    features,
                    key=lambda f: rf.feature_importances_[list(X.columns).index(f)]
                )

            representatives.append(best)
            for feat in features:
                if feat != best:
                    removed[feat] = f"correlated with {best} (cluster {cluster_id})"

        return representatives, cluster_importance

    def _apply_lasso(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        removed: Dict[str, str]
    ) -> Tuple[List[str], Dict[str, float]]:
        """Apply LASSO with cross-validation for embedded selection."""
        # Use Elastic Net for better handling of correlated features
        enet = ElasticNetCV(
            l1_ratio=0.5,
            cv=self.lasso_cv_folds,
            random_state=42,
            max_iter=10000
        )
        enet.fit(X, y)

        coefs = dict(zip(X.columns, enet.coef_))
        selected = [f for f, c in coefs.items() if abs(c) > 1e-6]

        for feat in X.columns:
            if feat not in selected:
                removed[feat] = f"zero LASSO coefficient"

        return selected, coefs

    def _filter_by_stability(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        removed: Dict[str, str]
    ) -> Tuple[List[str], Dict[str, float]]:
        """Filter features by stability across walk-forward folds."""
        tscv = TimeSeriesSplit(n_splits=5)
        feature_counts = defaultdict(int)

        for train_idx, val_idx in tscv.split(X):
            # Simple importance calculation per fold
            rf = RandomForestClassifier(n_estimators=30, max_depth=4, random_state=42)
            rf.fit(X.iloc[train_idx], y.iloc[train_idx])

            # Get top 80% of features by importance
            importance = pd.Series(rf.feature_importances_, index=X.columns)
            threshold = importance.quantile(0.2)
            selected = importance[importance >= threshold].index.tolist()

            for feat in selected:
                feature_counts[feat] += 1

        # Calculate stability scores
        n_folds = 5
        stability = {f: count / n_folds for f, count in feature_counts.items()}

        # Keep features above stability threshold
        stable = [f for f, s in stability.items() if s >= self.stability_threshold]

        for feat in X.columns:
            if feat not in stable:
                removed[feat] = f"unstable (stability={stability.get(feat, 0):.2f})"

        return stable, stability

    def _check_vif(
        self,
        X: pd.DataFrame,
        removed: Dict[str, str]
    ) -> Tuple[List[str], Dict[str, float]]:
        """Check and filter by VIF."""
        from statsmodels.stats.outliers_influence import variance_inflation_factor

        features = X.columns.tolist()
        vif_scores = {}

        while len(features) > 1:
            X_subset = X[features]

            # Calculate VIF for all features
            vif_values = []
            for i in range(len(features)):
                try:
                    vif = variance_inflation_factor(X_subset.values, i)
                    vif_values.append((features[i], vif))
                except:
                    vif_values.append((features[i], 1.0))

            # Find max VIF
            vif_df = pd.DataFrame(vif_values, columns=['feature', 'vif'])
            max_vif_row = vif_df.loc[vif_df['vif'].idxmax()]

            if max_vif_row['vif'] <= self.max_vif:
                # All VIFs acceptable
                for feat, vif in vif_values:
                    vif_scores[feat] = vif
                break

            # Remove feature with highest VIF
            removed[max_vif_row['feature']] = f"high VIF ({max_vif_row['vif']:.1f})"
            features.remove(max_vif_row['feature'])

        return features, vif_scores
```

### 9.2 Integration with Existing Pipeline

```python
# In src/phase1/stages/feature_selection/enhanced_selector.py

from src.phase1.utils.feature_selection import FEATURE_PRIORITY

def run_enhanced_feature_selection(
    df: pd.DataFrame,
    target_col: str = 'label_h10',
    output_path: Path = None
) -> EnhancedFeatureSelectionResult:
    """
    Run enhanced feature selection on Phase 1 output.
    """
    # Identify feature columns
    feature_cols = [c for c in df.columns
                    if c not in METADATA_COLUMNS
                    and not c.startswith(('label_', 'quality_', 'sample_weight_'))]

    X = df[feature_cols]
    y = df[target_col]

    selector = EnhancedFeatureSelector(
        variance_threshold=0.01,
        correlation_threshold=0.80,
        stability_threshold=0.6,
        max_vif=10.0,
    )

    result = selector.fit(X, y, feature_priority=FEATURE_PRIORITY)

    if output_path:
        save_selection_report(result, output_path)

    return result
```

---

## 10. Sources

### Academic Papers and Books

1. Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
2. Lopez de Prado, M. (2020). [Clustered Feature Importance (Presentation Slides)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3517595). SSRN.
3. Man, X., & Chan, E. (2021). [Cluster-based Feature Selection](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3880641). SSRN.

### Feature Selection Research (2024)

4. [Feature selection with annealing for forecasting financial time series](https://jfin-swufe.springeropen.com/articles/10.1186/s40854-024-00617-3). Financial Innovation, 2024.
5. [Survey of feature selection and extraction techniques for stock market prediction](https://link.springer.com/article/10.1186/s40854-022-00441-7). Financial Innovation, 2022.

### Cross-Validation and Time Series

6. [Cross Validation in Finance: Purging, Embargoing, Combinatorial](https://blog.quantinsti.com/cross-validation-embargo-purging-combinatorial/). QuantInsti.
7. [Purged Cross-Validation](https://en.wikipedia.org/wiki/Purged_cross-validation). Wikipedia.
8. [timeseriescv Library](https://github.com/sam31415/timeseriescv). GitHub.

### Multicollinearity and VIF

9. [Thresholds for Detecting Multicollinearity](https://stataiml.com/posts/60_multicollinearity_threshold_ml/). StatAIML.
10. [Variance Inflation Factor: How to Detect Multicollinearity](https://www.datacamp.com/tutorial/variance-inflation-factor). DataCamp.

### SHAP and Interpretability

11. [Interpreting financial time series with SHAP values](https://dl.acm.org/doi/10.5555/3370272.3370290). ACM, 2019.
12. [SHAP for Time Series Event Detection](https://towardsdatascience.com/shap-for-time-series-event-detection-5b4d9d0f96f4/). Towards Data Science.

### Wrapper Methods

13. [Recursive Feature Elimination (RFE) Made Simple](https://spotintelligence.com/2024/11/18/recursive-feature-elimination-rfe/). Spot Intelligence, 2024.
14. [Enhancing financial product forecasting accuracy using EMD and feature selection](https://www.sciencedirect.com/science/article/pii/S2199853125000666). ScienceDirect, 2025.

### Library Documentation

15. [MLFinLab: Clustered MDA and MDI](https://www.mlfinlab.com/en/latest/feature_importance/clustered.html). Hudson & Thames.
16. [MLFinLab: Feature Clustering](https://www.mlfinlab.com/en/latest/clustering/feature_clusters.html). Hudson & Thames.
17. [skfolio: CombinatorialPurgedCV](https://skfolio.org/generated/skfolio.model_selection.CombinatorialPurgedCV.html). skfolio.

### Mutual Information

18. [Mutual information for feature selection](https://joaodmrodrigues.github.io/elements-financial-machine-learning/information%20theory/mutual%20information/feature%20selection/feature%20importance/2021/02/06/mutal_information_and_feature_selection.html). Elements of Financial ML.

### Rolling Window and Stability

19. [Rolling Window Selection for Out-of-Sample Forecasting](https://www.sciencedirect.com/science/article/abs/pii/S0304407616301713). Journal of Econometrics.

### Overfitting and Sample Size

20. [The Curse of Dimensionality in Classification](https://www.visiondummy.com/2014/04/curse-dimensionality-affect-classification/). VisionDummy.
21. [How do you attack a machine learning problem with a large number of features?](https://sebastianraschka.com/faq/docs/large-num-features.html). Sebastian Raschka.

---

## Appendix A: Quick Reference Card

### Recommended Thresholds

| Parameter | Value | Notes |
|-----------|-------|-------|
| Variance threshold | 0.01 | Remove near-constant |
| Correlation threshold | 0.80 | Spearman, not Pearson |
| VIF threshold | 10.0 | Conservative |
| Stability threshold | 0.60 | 60% of folds |
| Sample:Feature ratio | 10:1 min | 20:1 preferred |

### Feature Priority (Top 10)

1. `log_return` (100)
2. `bb_position` (90)
3. `macd_hist` (90)
4. `rsi` (90)
5. `high_low_range` (90)
6. `volume_ratio` (85)
7. `adx` (85)
8. `close_to_vwap` (85)
9. `atr_7_pct` (85)
10. `vol_regime` (85)

### Pipeline Stages Checklist

- [ ] Variance filter (var < 0.01)
- [ ] Hierarchical clustering (corr > 0.80)
- [ ] Cluster representative selection
- [ ] LASSO/Elastic Net (embedded)
- [ ] Walk-forward stability check
- [ ] VIF verification (< 10)
- [ ] Sample:feature ratio check
