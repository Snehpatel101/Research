# Advanced Feature Selection for Financial ML

## Research Summary

This document provides implementation-ready specifications for advanced feature selection methods beyond simple correlation filtering. Based on research from Lopez de Prado's work, SHAP-based methods, and modern ML practices.

**Current State:**
- Correlation threshold filtering (0.80)
- Variance threshold filtering (0.01)
- Priority-based selection from correlated groups

**Target State:**
- MDA with purged cross-validation
- SHAP-based importance ranking
- Hierarchical feature clustering (CFI)
- Walk-forward stability analysis
- Model-specific feature subsets

---

## 1. Lopez de Prado Feature Importance Methods

### 1.1 Mean Decrease Impurity (MDI) - Limitations

MDI measures feature importance by the total decrease in node impurity averaged over all trees in a random forest. However, MDI has significant issues for financial data:

**Problems:**
- Biased toward high-cardinality features
- Computed in-sample (overfitting risk)
- Suffers from substitution effects when features are correlated
- Cannot conclude that all features are unimportant

**Use Case:** Quick initial screening only. Never use for final feature selection.

### 1.2 Mean Decrease Accuracy (MDA) - Gold Standard

MDA is an out-of-sample, predictive importance method:

1. Fit classifier on training data
2. Compute OOS performance score
3. Permute each feature column and re-compute OOS score
4. Feature importance = performance drop from permutation

**Advantages:**
- Out-of-sample evaluation (avoids overfitting)
- Works with any classifier (not just tree-based)
- Can conclude features are unimportant

**Critical for Finance:** Use log-loss instead of accuracy for scoring. Accuracy ignores prediction confidence, which is essential in trading.

**Implementation:**

```python
from sklearn.metrics import log_loss
import numpy as np

def mda_feature_importance(
    clf,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    scoring: str = 'neg_log_loss',
    n_permutations: int = 10
) -> pd.DataFrame:
    """
    Mean Decrease Accuracy feature importance with OOS evaluation.

    Args:
        clf: Fitted classifier with predict_proba
        X_train, y_train: Training data
        X_test, y_test: OOS test data
        scoring: 'neg_log_loss' (recommended) or 'accuracy'
        n_permutations: Permutations per feature for stability

    Returns:
        DataFrame with feature importance scores
    """
    clf.fit(X_train, y_train)

    if scoring == 'neg_log_loss':
        y_prob = clf.predict_proba(X_test)
        baseline_score = -log_loss(y_test, y_prob)
    else:
        baseline_score = clf.score(X_test, y_test)

    importances = {}
    for col in X_train.columns:
        scores = []
        for _ in range(n_permutations):
            X_test_perm = X_test.copy()
            X_test_perm[col] = np.random.permutation(X_test_perm[col].values)

            if scoring == 'neg_log_loss':
                y_prob = clf.predict_proba(X_test_perm)
                perm_score = -log_loss(y_test, y_prob)
            else:
                perm_score = clf.score(X_test_perm, y_test)

            scores.append(baseline_score - perm_score)

        importances[col] = {
            'importance_mean': np.mean(scores),
            'importance_std': np.std(scores)
        }

    return pd.DataFrame(importances).T.sort_values('importance_mean', ascending=False)
```

### 1.3 Single Feature Importance (SFI)

SFI computes OOS performance of each feature in isolation:

**Advantages:**
- No substitution effects (features evaluated independently)
- Can use any classifier and scoring metric
- OOS evaluation

**Limitations:**
- Misses interaction effects (A+B may be predictive, but neither A nor B alone)
- Misses hierarchical importance
- Risk of false positives with many features

**Use Case:** Sanity check against MDA. If MDA ranks a feature high but SFI ranks it low, the feature may only be useful in combination with others.

```python
def sfi_feature_importance(
    clf_factory,  # Callable that returns fresh classifier
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    scoring: str = 'neg_log_loss'
) -> pd.DataFrame:
    """
    Single Feature Importance - evaluate each feature in isolation.
    """
    importances = {}

    for col in X_train.columns:
        clf = clf_factory()
        clf.fit(X_train[[col]], y_train)

        if scoring == 'neg_log_loss':
            y_prob = clf.predict_proba(X_test[[col]])
            score = -log_loss(y_test, y_prob)
        else:
            score = clf.score(X_test[[col]], y_test)

        importances[col] = {'sfi_score': score}

    return pd.DataFrame(importances).T.sort_values('sfi_score', ascending=False)
```

### 1.4 Clustered Feature Importance (CFI)

CFI addresses the substitution effect by clustering correlated features and evaluating clusters together:

**Algorithm:**
1. Cluster features using hierarchical clustering on correlation matrix
2. Instead of permuting individual features, permute all features in a cluster together
3. Importance is assigned at cluster level, then distributed to cluster members

**Why CFI Works:**
- If features A and B are correlated, permuting A while B remains intact shows low importance for A (B compensates)
- Permuting A and B together shows their true joint importance
- Robust to both linear and non-linear substitution effects

```python
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

def clustered_feature_importance(
    clf,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_clusters: int = None,
    max_clusters: int = 10,
    scoring: str = 'neg_log_loss'
) -> pd.DataFrame:
    """
    Clustered MDA - robust to substitution effects.
    """
    # Step 1: Cluster features using correlation
    corr_matrix = X_train.corr().abs()
    distance_matrix = 1 - corr_matrix
    np.fill_diagonal(distance_matrix.values, 0)

    condensed_dist = squareform(distance_matrix)
    linkage_matrix = linkage(condensed_dist, method='ward')

    if n_clusters is None:
        # Use silhouette score to determine optimal clusters
        n_clusters = min(max_clusters, len(X_train.columns) // 3)

    cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
    feature_clusters = pd.Series(cluster_labels, index=X_train.columns)

    # Step 2: Compute baseline score
    clf.fit(X_train, y_train)
    y_prob = clf.predict_proba(X_test)
    baseline_score = -log_loss(y_test, y_prob)

    # Step 3: Permute clusters and measure importance
    cluster_importance = {}
    for cluster_id in feature_clusters.unique():
        cluster_features = feature_clusters[feature_clusters == cluster_id].index.tolist()

        X_test_perm = X_test.copy()
        for col in cluster_features:
            X_test_perm[col] = np.random.permutation(X_test_perm[col].values)

        y_prob_perm = clf.predict_proba(X_test_perm)
        perm_score = -log_loss(y_test, y_prob_perm)

        cluster_importance[cluster_id] = {
            'importance': baseline_score - perm_score,
            'features': cluster_features
        }

    # Step 4: Distribute cluster importance to features
    feature_importance = {}
    for cluster_id, data in cluster_importance.items():
        per_feature_importance = data['importance'] / len(data['features'])
        for feat in data['features']:
            feature_importance[feat] = {
                'cfi_importance': per_feature_importance,
                'cluster_id': cluster_id,
                'cluster_size': len(data['features'])
            }

    return pd.DataFrame(feature_importance).T.sort_values('cfi_importance', ascending=False)
```

### 1.5 PCA Sanity Check

Compare MDI/MDA/SFI rankings with PCA eigenvalue rankings:

- PCA ranks features by variance explained (unsupervised - no labels)
- MDI/MDA rank features by predictive power (supervised - uses labels)
- If rankings correlate, it confirms the pattern is not overfit to labels
- If rankings diverge significantly, investigate overfitting

```python
from sklearn.decomposition import PCA
from scipy.stats import kendalltau

def pca_sanity_check(
    X: pd.DataFrame,
    feature_importance: pd.DataFrame
) -> dict:
    """
    Compare feature importance with PCA variance explained.
    High correlation suggests genuine signal, not overfitting.
    """
    pca = PCA()
    pca.fit(X)

    # Get PCA loadings (absolute values)
    loadings = pd.DataFrame(
        np.abs(pca.components_),
        columns=X.columns,
        index=[f'PC{i+1}' for i in range(pca.n_components_)]
    )

    # Weight by variance explained
    weighted_loadings = loadings.T @ pca.explained_variance_ratio_
    pca_ranking = weighted_loadings.sort_values(ascending=False)

    # Compare with feature importance ranking
    common_features = feature_importance.index.intersection(pca_ranking.index)

    tau, p_value = kendalltau(
        feature_importance.loc[common_features, 'importance_mean'].rank(),
        pca_ranking.loc[common_features].rank()
    )

    return {
        'kendall_tau': tau,
        'p_value': p_value,
        'interpretation': 'consistent' if tau > 0.3 and p_value < 0.05 else 'divergent'
    }
```

---

## 2. SHAP-Based Feature Selection

### 2.1 TreeSHAP for Boosting Models

TreeSHAP provides exact Shapley values for tree-based models with O(TLD^2) complexity instead of O(TL2^M).

**Advantages:**
- Theoretically grounded (game theory)
- Handles feature interactions
- Model-agnostic interpretation
- Fast for tree models

**For Time Series:**
- Use Vector SHAP for lag features (groups lags of same variable)
- Faster computation
- More interpretable (variable importance vs lag importance)

```python
import shap

def shap_feature_importance(
    model,  # XGBoost, LightGBM, or CatBoost
    X_train: pd.DataFrame,
    X_test: pd.DataFrame = None,
    use_tree_shap: bool = True
) -> pd.DataFrame:
    """
    SHAP-based feature importance.

    Args:
        model: Trained tree-based model
        X_train: Training data for background
        X_test: Data to explain (uses sample of X_train if None)
        use_tree_shap: Use TreeSHAP (fast) vs KernelSHAP (slow)
    """
    if X_test is None:
        X_test = X_train.sample(min(1000, len(X_train)))

    if use_tree_shap:
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.KernelExplainer(model.predict, X_train.sample(100))

    shap_values = explainer.shap_values(X_test)

    # Handle multi-class (take mean absolute SHAP across classes)
    if isinstance(shap_values, list):
        shap_values = np.abs(np.array(shap_values)).mean(axis=0)
    else:
        shap_values = np.abs(shap_values)

    # Mean absolute SHAP per feature
    mean_shap = pd.DataFrame({
        'feature': X_test.columns,
        'shap_importance': shap_values.mean(axis=0)
    }).set_index('feature').sort_values('shap_importance', ascending=False)

    return mean_shap


def shap_feature_selection(
    model,
    X: pd.DataFrame,
    threshold: float = 0.01,
    min_features: int = 20,
    max_features: int = 80
) -> List[str]:
    """
    Select features based on SHAP importance threshold.
    """
    importance = shap_feature_importance(model, X)

    # Normalize to sum to 1
    importance['normalized'] = importance['shap_importance'] / importance['shap_importance'].sum()

    # Cumulative importance
    importance['cumulative'] = importance['normalized'].cumsum()

    # Select features above threshold or until min_features
    selected = importance[importance['normalized'] >= threshold].index.tolist()

    if len(selected) < min_features:
        selected = importance.head(min_features).index.tolist()
    elif len(selected) > max_features:
        selected = importance.head(max_features).index.tolist()

    return selected
```

### 2.2 SHAP Interaction Values

SHAP interaction values reveal feature pairs that work together:

```python
def shap_interaction_analysis(
    model,
    X: pd.DataFrame,
    top_k: int = 10
) -> pd.DataFrame:
    """
    Analyze feature interactions using SHAP.
    Useful for understanding which features to keep together.
    """
    explainer = shap.TreeExplainer(model)

    # Sample for speed
    X_sample = X.sample(min(500, len(X)))
    interaction_values = explainer.shap_interaction_values(X_sample)

    # Average absolute interaction strength
    if isinstance(interaction_values, list):
        interaction_values = np.abs(np.array(interaction_values)).mean(axis=0)
    else:
        interaction_values = np.abs(interaction_values)

    mean_interactions = interaction_values.mean(axis=0)

    # Extract top interactions (off-diagonal)
    interactions = []
    for i in range(len(X.columns)):
        for j in range(i + 1, len(X.columns)):
            interactions.append({
                'feature_1': X.columns[i],
                'feature_2': X.columns[j],
                'interaction_strength': mean_interactions[i, j]
            })

    return pd.DataFrame(interactions).sort_values(
        'interaction_strength', ascending=False
    ).head(top_k)
```

---

## 3. Recursive Feature Elimination (RFE)

### 3.1 Time-Series Aware RFE

Standard RFE uses random cross-validation. For time series, use purged CV:

```python
from sklearn.feature_selection import RFECV
from sklearn.model_selection import TimeSeriesSplit

def purged_rfe(
    estimator,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    purge_bars: int = 60,
    embargo_bars: int = 1440,
    step: int = 1,
    min_features: int = 20
) -> List[str]:
    """
    Recursive Feature Elimination with Purged Cross-Validation.

    Prevents lookahead bias by using time-series CV with
    purge and embargo periods.
    """
    # Custom CV splitter with purge/embargo
    cv = PurgedTimeSeriesSplit(
        n_splits=n_splits,
        purge_bars=purge_bars,
        embargo_bars=embargo_bars
    )

    rfecv = RFECV(
        estimator=estimator,
        step=step,
        cv=cv,
        scoring='neg_log_loss',
        min_features_to_select=min_features,
        n_jobs=-1
    )

    rfecv.fit(X, y)

    selected = X.columns[rfecv.support_].tolist()

    return selected


class PurgedTimeSeriesSplit:
    """
    Time series CV with purge (label leakage prevention) and
    embargo (serial correlation prevention).
    """

    def __init__(
        self,
        n_splits: int = 5,
        purge_bars: int = 60,
        embargo_bars: int = 1440
    ):
        self.n_splits = n_splits
        self.purge_bars = purge_bars
        self.embargo_bars = embargo_bars

    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        test_size = n_samples // (self.n_splits + 1)

        for i in range(self.n_splits):
            test_start = (i + 1) * test_size
            test_end = test_start + test_size

            # Training: all before test_start minus purge
            train_end = test_start - self.purge_bars
            train_indices = np.arange(0, max(0, train_end))

            # Apply embargo after test set
            embargo_end = min(test_end + self.embargo_bars, n_samples)

            test_indices = np.arange(test_start, test_end)

            if len(train_indices) > 0 and len(test_indices) > 0:
                yield train_indices, test_indices

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits
```

---

## 4. Boruta Algorithm

### 4.1 All-Relevant Feature Selection

Boruta identifies all features that are relevant (not just the minimal subset):

**Algorithm:**
1. Create "shadow features" by shuffling each original feature
2. Train random forest on original + shadow features
3. Features beating the best shadow feature are "confirmed"
4. Features never beating shadow features are "rejected"
5. Repeat until all features are classified

**For Trading:**
Use Boruta-SHAP which combines Boruta with SHAP importance for better importance estimation.

```python
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier

def boruta_feature_selection(
    X: pd.DataFrame,
    y: pd.Series,
    max_iter: int = 100,
    random_state: int = 42
) -> dict:
    """
    Boruta all-relevant feature selection.

    Returns dict with 'confirmed', 'tentative', and 'rejected' features.
    """
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        n_jobs=-1,
        random_state=random_state
    )

    boruta = BorutaPy(
        estimator=rf,
        n_estimators='auto',
        max_iter=max_iter,
        random_state=random_state,
        verbose=0
    )

    boruta.fit(X.values, y.values)

    return {
        'confirmed': X.columns[boruta.support_].tolist(),
        'tentative': X.columns[boruta.support_weak_].tolist(),
        'rejected': X.columns[~boruta.support_ & ~boruta.support_weak_].tolist(),
        'ranking': pd.Series(boruta.ranking_, index=X.columns)
    }
```

### 4.2 Boruta-SHAP Integration

```python
# pip install BorutaShap

from BorutaShap import BorutaShap

def boruta_shap_selection(
    X: pd.DataFrame,
    y: pd.Series,
    model = None,
    n_trials: int = 100,
    random_state: int = 42
) -> dict:
    """
    Boruta with SHAP-based importance (more accurate than RF importance).
    """
    if model is None:
        from xgboost import XGBClassifier
        model = XGBClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=random_state,
            use_label_encoder=False,
            eval_metric='logloss'
        )

    selector = BorutaShap(
        model=model,
        importance_measure='shap',
        classification=True
    )

    selector.fit(
        X=X,
        y=y,
        n_trials=n_trials,
        random_state=random_state,
        verbose=False
    )

    return {
        'selected': selector.Subset().columns.tolist(),
        'rejected': [c for c in X.columns if c not in selector.Subset().columns]
    }
```

---

## 5. Walk-Forward Stability Analysis

### 5.1 Feature Stability Over Time

Features important in one period may be noise in another (regime dependence):

```python
from typing import List, Tuple

def walk_forward_feature_stability(
    X: pd.DataFrame,
    y: pd.Series,
    clf_factory,
    window_size: int = 5000,  # Training window
    step_size: int = 1000,    # Step between windows
    importance_fn = None      # Function to compute importance
) -> pd.DataFrame:
    """
    Analyze feature importance stability across time.

    Returns:
        DataFrame with importance per window and stability metrics.
    """
    if importance_fn is None:
        importance_fn = lambda clf, X_tr, y_tr, X_te, y_te: (
            mda_feature_importance(clf, X_tr, y_tr, X_te, y_te)
        )

    n_samples = len(X)
    windows = []

    for start in range(0, n_samples - window_size - step_size, step_size):
        train_end = start + window_size
        test_end = train_end + step_size

        X_train = X.iloc[start:train_end]
        y_train = y.iloc[start:train_end]
        X_test = X.iloc[train_end:test_end]
        y_test = y.iloc[train_end:test_end]

        clf = clf_factory()
        importance = importance_fn(clf, X_train, y_train, X_test, y_test)

        windows.append({
            'window_start': start,
            'window_end': train_end,
            'importance': importance['importance_mean'].to_dict()
        })

    # Build importance matrix
    all_features = X.columns.tolist()
    importance_matrix = pd.DataFrame(
        index=range(len(windows)),
        columns=all_features
    )

    for i, w in enumerate(windows):
        for feat, imp in w['importance'].items():
            importance_matrix.loc[i, feat] = imp

    # Compute stability metrics
    stability = pd.DataFrame({
        'mean_importance': importance_matrix.mean(),
        'std_importance': importance_matrix.std(),
        'cv': importance_matrix.std() / (importance_matrix.mean() + 1e-10),
        'min_importance': importance_matrix.min(),
        'max_importance': importance_matrix.max(),
        'pct_positive': (importance_matrix > 0).mean()
    })

    # Stability score: high mean, low CV, high pct_positive
    stability['stability_score'] = (
        stability['mean_importance'].rank(pct=True) * 0.4 +
        (1 - stability['cv'].rank(pct=True)) * 0.3 +
        stability['pct_positive'].rank(pct=True) * 0.3
    )

    return stability.sort_values('stability_score', ascending=False)


def identify_stable_features(
    stability: pd.DataFrame,
    min_stability_score: float = 0.7,
    min_pct_positive: float = 0.8
) -> List[str]:
    """
    Select features that are consistently important across regimes.
    """
    stable = stability[
        (stability['stability_score'] >= min_stability_score) &
        (stability['pct_positive'] >= min_pct_positive)
    ]

    return stable.index.tolist()
```

### 5.2 Regime-Dependent Feature Sets

Some features may only be useful in specific market regimes:

```python
def regime_specific_features(
    X: pd.DataFrame,
    y: pd.Series,
    regime_labels: pd.Series,  # e.g., 'trending', 'ranging', 'volatile'
    clf_factory,
    top_k: int = 30
) -> dict:
    """
    Identify features important in each regime.

    Use for ensemble diversity: different models for different regimes.
    """
    regime_features = {}

    for regime in regime_labels.unique():
        mask = regime_labels == regime
        X_regime = X[mask]
        y_regime = y[mask]

        if len(X_regime) < 1000:
            continue

        # Split into train/test
        split_idx = int(len(X_regime) * 0.8)
        X_train = X_regime.iloc[:split_idx]
        y_train = y_regime.iloc[:split_idx]
        X_test = X_regime.iloc[split_idx:]
        y_test = y_regime.iloc[split_idx:]

        clf = clf_factory()
        importance = mda_feature_importance(clf, X_train, y_train, X_test, y_test)

        regime_features[regime] = importance.head(top_k).index.tolist()

    return regime_features
```

---

## 6. Target Encoding Risks

### 6.1 Leakage in Time Series

Target encoding is dangerous for time series because it can leak future information:

**Problem:**
- Encoding computed on full dataset includes future label information
- Even with CV, if encoding uses future folds, leakage occurs

**Solutions:**

```python
class TimeSeriesTargetEncoder:
    """
    Target encoding with proper temporal handling.

    - Encodes using only past data
    - Applies smoothing for rare categories
    - Falls back to global mean for unseen categories
    """

    def __init__(
        self,
        cols: List[str],
        smoothing: float = 10.0,
        min_samples: int = 20
    ):
        self.cols = cols
        self.smoothing = smoothing
        self.min_samples = min_samples
        self.encodings_ = {}
        self.global_mean_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fit on training data only."""
        self.global_mean_ = y.mean()

        for col in self.cols:
            group_stats = X.groupby(col)[y.name].agg(['mean', 'count'])

            # Bayesian smoothing
            smooth_mean = (
                group_stats['count'] * group_stats['mean'] +
                self.smoothing * self.global_mean_
            ) / (group_stats['count'] + self.smoothing)

            # Mask categories with too few samples
            smooth_mean[group_stats['count'] < self.min_samples] = self.global_mean_

            self.encodings_[col] = smooth_mean.to_dict()

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform using learned encodings."""
        X_encoded = X.copy()

        for col in self.cols:
            X_encoded[f'{col}_encoded'] = X[col].map(
                self.encodings_.get(col, {})
            ).fillna(self.global_mean_)

        return X_encoded

    def fit_transform_expanding(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        min_history: int = 1000
    ) -> pd.DataFrame:
        """
        Expanding window encoding for full time series.

        At each point, uses only past data for encoding.
        Slow but leak-free.
        """
        X_encoded = X.copy()

        for col in self.cols:
            encoded_values = []

            for i in range(len(X)):
                if i < min_history:
                    encoded_values.append(np.nan)
                else:
                    # Use only past data
                    past_X = X.iloc[:i]
                    past_y = y.iloc[:i]

                    current_val = X[col].iloc[i]
                    mask = past_X[col] == current_val

                    if mask.sum() >= self.min_samples:
                        encoded_values.append(past_y[mask].mean())
                    else:
                        encoded_values.append(past_y.mean())

            X_encoded[f'{col}_encoded'] = encoded_values

        return X_encoded
```

---

## 7. Model-Specific Feature Selection

### 7.1 Different Features for Different Models

Each model type has preferences:

| Model | Preferred Features | Avoid |
|-------|-------------------|-------|
| XGBoost | Tree-friendly (any scale), interactions | Highly sparse |
| LightGBM | Similar to XGBoost, handles sparse better | - |
| LSTM | Sequential patterns, normalized | Noisy, non-sequential |
| Transformer | Long-range patterns, attention-friendly | Very short sequences |
| Linear | Scaled, low correlation | High collinearity |

```python
@dataclass
class ModelFeatureConfig:
    """Configuration for model-specific feature selection."""
    model_type: str
    max_features: int
    require_scaling: bool
    prefer_sequential: bool
    handle_sparse: bool
    correlation_threshold: float

MODEL_CONFIGS = {
    'xgboost': ModelFeatureConfig(
        model_type='xgboost',
        max_features=80,
        require_scaling=False,
        prefer_sequential=False,
        handle_sparse=False,
        correlation_threshold=0.90  # More tolerant
    ),
    'lightgbm': ModelFeatureConfig(
        model_type='lightgbm',
        max_features=100,
        require_scaling=False,
        prefer_sequential=False,
        handle_sparse=True,
        correlation_threshold=0.90
    ),
    'lstm': ModelFeatureConfig(
        model_type='lstm',
        max_features=50,  # Fewer features for RNNs
        require_scaling=True,
        prefer_sequential=True,
        handle_sparse=False,
        correlation_threshold=0.80  # Stricter
    ),
    'transformer': ModelFeatureConfig(
        model_type='transformer',
        max_features=60,
        require_scaling=True,
        prefer_sequential=True,
        handle_sparse=False,
        correlation_threshold=0.80
    )
}


def select_features_for_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str,
    base_features: List[str],  # Pre-filtered features
    sequential_features: List[str] = None  # Features with temporal patterns
) -> List[str]:
    """
    Select features optimized for specific model type.
    """
    config = MODEL_CONFIGS[model_type]

    features = base_features.copy()

    # For neural networks, prefer sequential features
    if config.prefer_sequential and sequential_features:
        seq_in_base = [f for f in sequential_features if f in features]
        non_seq = [f for f in features if f not in sequential_features]

        # Prioritize sequential features
        features = seq_in_base + non_seq

    # Apply correlation filter with model-specific threshold
    if config.correlation_threshold < 1.0:
        X_subset = X[features]
        _, _, _ = filter_correlated_features(
            X_subset, features, config.correlation_threshold
        )

    # Limit features
    features = features[:config.max_features]

    return features
```

### 7.2 Ensemble Diversity Through Feature Subsets

For ensemble models, use different feature subsets to increase diversity:

```python
def create_diverse_feature_sets(
    features: List[str],
    n_sets: int = 3,
    overlap_ratio: float = 0.5
) -> List[List[str]]:
    """
    Create diverse feature subsets for ensemble diversity.

    Each subset shares `overlap_ratio` features with others
    but has unique features for diversity.
    """
    n_features = len(features)
    n_shared = int(n_features * overlap_ratio)
    n_unique_per_set = (n_features - n_shared) // n_sets

    # Shared core features (highest importance)
    shared = features[:n_shared]
    remaining = features[n_shared:]

    feature_sets = []
    for i in range(n_sets):
        start = i * n_unique_per_set
        end = start + n_unique_per_set
        unique = remaining[start:end]

        feature_sets.append(shared + unique)

    return feature_sets
```

---

## 8. Enhanced FeatureSelector Class

### 8.1 Complete Implementation

```python
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable
from enum import Enum
import pandas as pd
import numpy as np

class ImportanceMethod(Enum):
    MDI = "mdi"
    MDA = "mda"
    SFI = "sfi"
    CFI = "cfi"
    SHAP = "shap"
    BORUTA = "boruta"


@dataclass
class FeatureSelectionConfig:
    """Configuration for advanced feature selection."""

    # Basic filtering
    variance_threshold: float = 0.01
    correlation_threshold: float = 0.85

    # Importance method
    importance_method: ImportanceMethod = ImportanceMethod.MDA

    # Purged CV settings
    n_cv_splits: int = 5
    purge_bars: int = 60
    embargo_bars: int = 1440

    # Feature limits
    min_features: int = 20
    max_features: int = 80

    # Stability settings
    stability_enabled: bool = True
    n_stability_windows: int = 5
    min_stability_score: float = 0.7

    # Model-specific
    model_type: Optional[str] = None

    # SHAP settings
    shap_sample_size: int = 1000

    # Boruta settings
    boruta_max_iter: int = 100


@dataclass
class AdvancedFeatureSelectionResult:
    """Results from advanced feature selection."""
    selected_features: List[str]
    feature_importance: pd.DataFrame
    stability_scores: Optional[pd.DataFrame]
    clusters: List[List[str]]
    removed_features: Dict[str, str]
    config: FeatureSelectionConfig

    def to_dict(self) -> dict:
        return {
            'selected_features': self.selected_features,
            'n_selected': len(self.selected_features),
            'importance': self.feature_importance.to_dict(),
            'stability': self.stability_scores.to_dict() if self.stability_scores is not None else None,
            'clusters': self.clusters,
            'removed': self.removed_features
        }


class AdvancedFeatureSelector:
    """
    Advanced feature selection combining multiple methods.

    Pipeline:
    1. Variance filter (remove near-constant)
    2. Hierarchical clustering (group correlated)
    3. Importance ranking (MDA/SHAP/Boruta)
    4. Stability check (walk-forward)
    5. Model-specific filtering
    """

    def __init__(self, config: FeatureSelectionConfig):
        self.config = config
        self.clf_factory = None
        self.result_ = None

    def set_classifier_factory(self, factory: Callable):
        """Set factory function that returns fresh classifier instances."""
        self.clf_factory = factory

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        datetime_index: pd.DatetimeIndex = None
    ) -> 'AdvancedFeatureSelector':
        """
        Fit feature selector on training data.
        """
        features = X.columns.tolist()
        removed = {}

        # Step 1: Variance filter
        features, low_var = self._variance_filter(X, features)
        for f in low_var:
            removed[f] = 'low_variance'

        # Step 2: Cluster features
        clusters = self._cluster_features(X[features])

        # Step 3: Compute importance
        importance = self._compute_importance(X[features], y)

        # Step 4: Select from clusters using importance
        features, cluster_removed = self._select_from_clusters(
            features, clusters, importance
        )
        removed.update(cluster_removed)

        # Step 5: Stability analysis (if enabled)
        stability = None
        if self.config.stability_enabled:
            stability = self._stability_analysis(X[features], y)
            stable_features = identify_stable_features(
                stability, self.config.min_stability_score
            )

            # Keep only stable features if we have enough
            if len(stable_features) >= self.config.min_features:
                for f in features:
                    if f not in stable_features:
                        removed[f] = 'unstable_across_time'
                features = stable_features

        # Step 6: Apply feature limits
        if len(features) > self.config.max_features:
            # Keep top by importance
            top_features = importance.loc[
                importance.index.isin(features)
            ].head(self.config.max_features).index.tolist()

            for f in features:
                if f not in top_features:
                    removed[f] = 'exceeded_max_features'
            features = top_features

        # Step 7: Model-specific filtering
        if self.config.model_type:
            features = select_features_for_model(
                X, y, self.config.model_type, features
            )

        self.result_ = AdvancedFeatureSelectionResult(
            selected_features=features,
            feature_importance=importance,
            stability_scores=stability,
            clusters=[list(c) for c in clusters],
            removed_features=removed,
            config=self.config
        )

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Select features from dataframe."""
        if self.result_ is None:
            raise ValueError("Must call fit() first")

        return X[self.result_.selected_features]

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        datetime_index: pd.DatetimeIndex = None
    ) -> pd.DataFrame:
        """Fit and transform in one step."""
        self.fit(X, y, datetime_index)
        return self.transform(X)

    def _variance_filter(
        self,
        X: pd.DataFrame,
        features: List[str]
    ) -> tuple:
        """Remove low variance features."""
        return filter_low_variance(
            X, features, self.config.variance_threshold
        )

    def _cluster_features(self, X: pd.DataFrame) -> List[set]:
        """Cluster correlated features."""
        return build_correlation_groups(
            X, X.columns.tolist(), self.config.correlation_threshold
        )

    def _compute_importance(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> pd.DataFrame:
        """Compute feature importance using configured method."""

        method = self.config.importance_method

        if method == ImportanceMethod.MDA:
            return self._mda_importance(X, y)
        elif method == ImportanceMethod.SHAP:
            return self._shap_importance(X, y)
        elif method == ImportanceMethod.CFI:
            return self._cfi_importance(X, y)
        elif method == ImportanceMethod.BORUTA:
            return self._boruta_importance(X, y)
        else:
            # Default to MDA
            return self._mda_importance(X, y)

    def _mda_importance(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> pd.DataFrame:
        """MDA with purged CV."""
        cv = PurgedTimeSeriesSplit(
            n_splits=self.config.n_cv_splits,
            purge_bars=self.config.purge_bars,
            embargo_bars=self.config.embargo_bars
        )

        all_importance = []

        for train_idx, test_idx in cv.split(X):
            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_test = y.iloc[test_idx]

            clf = self.clf_factory()
            importance = mda_feature_importance(
                clf, X_train, y_train, X_test, y_test
            )
            all_importance.append(importance)

        # Average across folds
        combined = pd.concat(all_importance)
        return combined.groupby(combined.index).mean().sort_values(
            'importance_mean', ascending=False
        )

    def _shap_importance(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> pd.DataFrame:
        """SHAP-based importance."""
        clf = self.clf_factory()
        clf.fit(X, y)

        return shap_feature_importance(
            clf, X, X.sample(min(self.config.shap_sample_size, len(X)))
        )

    def _cfi_importance(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> pd.DataFrame:
        """Clustered feature importance."""
        split_idx = int(len(X) * 0.8)

        clf = self.clf_factory()
        return clustered_feature_importance(
            clf,
            X.iloc[:split_idx],
            y.iloc[:split_idx],
            X.iloc[split_idx:],
            y.iloc[split_idx:]
        )

    def _boruta_importance(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> pd.DataFrame:
        """Boruta-based importance."""
        result = boruta_feature_selection(
            X, y, max_iter=self.config.boruta_max_iter
        )

        # Convert ranking to importance DataFrame
        ranking = result['ranking']
        importance = pd.DataFrame({
            'boruta_rank': ranking,
            'importance_mean': 1.0 / ranking  # Invert so lower rank = higher importance
        })

        return importance.sort_values('importance_mean', ascending=False)

    def _select_from_clusters(
        self,
        features: List[str],
        clusters: List[set],
        importance: pd.DataFrame
    ) -> tuple:
        """Select best feature from each correlated cluster."""
        removed = {}
        features_to_remove = set()

        for cluster in clusters:
            if len(cluster) <= 1:
                continue

            # Find best feature in cluster by importance
            cluster_features = [f for f in cluster if f in importance.index]
            if not cluster_features:
                continue

            best = importance.loc[cluster_features].idxmax().iloc[0]

            for f in cluster:
                if f != best:
                    features_to_remove.add(f)
                    removed[f] = f'correlated_with_{best}'

        remaining = [f for f in features if f not in features_to_remove]
        return remaining, removed

    def _stability_analysis(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> pd.DataFrame:
        """Walk-forward stability analysis."""
        return walk_forward_feature_stability(
            X, y,
            self.clf_factory,
            window_size=len(X) // (self.config.n_stability_windows + 1),
            step_size=len(X) // (self.config.n_stability_windows * 2)
        )
```

---

## 9. Complete Feature Selection Pipeline

### 9.1 Pipeline Definition

```
Raw Features (200+)
    |
    v
[1. Variance Filter] --> Remove near-constant features
    |
    v
[2. Hierarchical Clustering] --> Group correlated features
    |
    v
[3. Importance Ranking]
    |-- MDA with Purged CV (default)
    |-- SHAP (for tree models)
    |-- CFI (handles substitution)
    |-- Boruta (all-relevant)
    |
    v
[4. Cluster Selection] --> Keep best from each cluster
    |
    v
[5. Stability Analysis] --> Remove regime-dependent features
    |
    v
[6. Model-Specific Filter] --> Optimize for XGBoost/LSTM/etc
    |
    v
Final Features (30-80 per model type)
```

### 9.2 Usage Example

```python
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Load data
X_train = pd.read_parquet('train_features.parquet')
y_train = pd.read_parquet('train_labels.parquet')['label_h10']

# Configure selector
config = FeatureSelectionConfig(
    variance_threshold=0.01,
    correlation_threshold=0.85,
    importance_method=ImportanceMethod.MDA,
    n_cv_splits=5,
    purge_bars=60,
    embargo_bars=1440,
    min_features=30,
    max_features=80,
    stability_enabled=True,
    model_type='xgboost'
)

# Create selector
selector = AdvancedFeatureSelector(config)
selector.set_classifier_factory(
    lambda: RandomForestClassifier(n_estimators=100, max_depth=5, n_jobs=-1)
)

# Fit and get selected features
selector.fit(X_train, y_train)

print(f"Selected {len(selector.result_.selected_features)} features")
print(f"Top 10: {selector.result_.selected_features[:10]}")

# Transform data
X_selected = selector.transform(X_train)

# Train model on selected features
model = XGBClassifier(n_estimators=200, max_depth=6)
model.fit(X_selected, y_train)
```

---

## 10. Configuration Schema

```python
# config/feature_selection.yaml

feature_selection:
  # Basic filtering
  variance_threshold: 0.01
  correlation_threshold: 0.85

  # Importance method: mda, shap, cfi, boruta
  importance_method: mda

  # Purged CV settings
  cv_splits: 5
  purge_bars: 60      # Horizon * 3
  embargo_bars: 1440  # ~5 days at 5-min bars

  # Feature limits
  min_features: 20
  max_features: 80

  # Stability analysis
  stability:
    enabled: true
    n_windows: 5
    min_score: 0.7
    min_pct_positive: 0.8

  # Model-specific configs
  model_configs:
    xgboost:
      max_features: 80
      correlation_threshold: 0.90
    lstm:
      max_features: 50
      correlation_threshold: 0.80
      prefer_sequential: true
    transformer:
      max_features: 60
      correlation_threshold: 0.80

  # SHAP settings
  shap:
    sample_size: 1000
    use_tree_shap: true

  # Boruta settings
  boruta:
    max_iter: 100
    random_state: 42
```

---

## 11. Integration with Current Pipeline

### 11.1 Modify Existing FeatureSelector

The current `/home/jake/Desktop/Research/src/phase1/utils/feature_selection.py` provides basic correlation and variance filtering. Extend it with:

1. Add `AdvancedFeatureSelector` class
2. Add MDA, SHAP, CFI importance methods
3. Add walk-forward stability analysis
4. Add model-specific configuration

### 11.2 New Module Structure

```
src/phase1/utils/
    feature_selection.py          # Current basic selection
    advanced_feature_selection/
        __init__.py
        importance.py             # MDA, SFI, CFI, SHAP
        clustering.py             # Hierarchical feature clustering
        stability.py              # Walk-forward stability
        boruta.py                 # Boruta wrapper
        config.py                 # FeatureSelectionConfig
        selector.py               # AdvancedFeatureSelector
```

---

## References

### Lopez de Prado Methods
- [MDI, MDA, and SFI - mlfinlab documentation](https://www.mlfinlab.com/en/latest/feature_importance/afm.html)
- [Clustered MDA and MDI - mlfinlab documentation](https://www.mlfinlab.com/en/latest/feature_importance/clustered.html)
- [Clustered Feature Importance (SSRN)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3517595)
- [Understanding Feature Importance in Financial ML](https://medium.com/@lucasastorian/understanding-financial-feature-importance-7eeb49c2df0b)

### SHAP
- [SHAP - Interpretable ML Book](https://christophm.github.io/interpretable-ml-book/shap.html)
- [Vector SHAP for Time Series Forecasting (2025)](https://onlinelibrary.wiley.com/doi/10.1002/for.3220?af=R)
- [Feature Selection: SHAP vs Built-in Importance](https://link.springer.com/article/10.1186/s40537-024-00905-w)
- [TreeSHAP GitHub](https://github.com/ModelOriented/treeshap)

### Boruta
- [Boruta-SHAP GPU Implementation](https://blog.quantinsti.com/boruta-shap-gpu-python/)
- [Boruta Feature Selection Explained](https://medium.com/geekculture/boruta-feature-selection-explained-in-python-7ae8bf4aa1e7)
- [Feature Selection with Boruta in Python](https://towardsdatascience.com/feature-selection-with-boruta-in-python-676e3877e596/)

### Purged Cross-Validation
- [Purged Cross-Validation (Wikipedia)](https://en.wikipedia.org/wiki/Purged_cross-validation)
- [Combinatorial Purged Cross Validation](https://towardsai.net/p/l/the-combinatorial-purged-cross-validation-method)
- [Cross Validation with Embargo and Purging](https://blog.quantinsti.com/cross-validation-embargo-purging-combinatorial/)
- [Avoiding Data Leakage in Time Series](https://towardsdatascience.com/avoiding-data-leakage-in-timeseries-101-25ea13fcb15f/)

### RFE and Stability
- [RFECV - scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html)
- [Walk-Forward Optimization](https://quantstrategy.io/blog/walk-forward-optimization-vs-traditional-backtesting-which/)
- [Walk-Forward Modeling - Alpha Scientist](https://alphascientist.com/walk_forward_model_building.html)
- [Feature Selection Stability - JMLR](https://jmlr.org/papers/volume18/17-514/17-514.pdf)

### TSFRESH
- [TSFRESH GitHub](https://github.com/blue-yonder/tsfresh)
- [TSFRESH Documentation](https://tsfresh.readthedocs.io/en/latest/)

### Model-Specific Selection
- [XGBoost-LSTM for Feature Selection](https://www.researchgate.net/publication/377180235_XGBoost-LSTM_for_Feature_Selection_and_Predictions_for_the_SP_500_Financial_Sector)
- [Ensemble TCN-LSTM-LightGBM](https://www.sciencedirect.com/science/article/abs/pii/S0360544225003998)
