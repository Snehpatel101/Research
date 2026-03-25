"""
Regime-conditional feature selection.

Computes feature importance per market regime (low-vol vs high-vol).
Features that are important in ANY regime are included via a union approach.
This allows models to leverage regime-specific signals: e.g. trend features
during trending periods and mean-reversion features during sideways markets.

Reference: Phase E5 of Feature Governance roadmap.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

logger = logging.getLogger(__name__)


def compute_regime_importance(
    df: pd.DataFrame,
    feature_names: list[str],
    label_col: str,
    n_regimes: int = 2,
    min_samples_per_regime: int = 200,
    n_estimators: int = 50,
    n_repeats: int = 3,
    random_state: int = 42,
) -> tuple[pd.Series, dict[int, pd.Series]] | None:
    """Compute feature importance per market regime.

    Uses volatility regime detection to split data into regime subsets,
    then runs MDA (permutation importance) within each regime. The combined
    importance is the maximum across regimes for each feature (union approach).

    Args:
        df: DataFrame containing features + label column + OHLCV for regime detection.
        feature_names: Feature column names.
        label_col: Name of the label column.
        n_regimes: Number of regimes (2 = low/high vol, 3 = low/medium/high).
        min_samples_per_regime: Minimum rows per regime to run MDA. Regimes
            with fewer samples are skipped.
        n_estimators: Number of trees for RF importance estimator.
        n_repeats: Number of permutation repeats for MDA.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (combined_importance, per_regime_importances) or None if
        regime detection fails or all regimes have insufficient samples.

        - combined_importance: pd.Series with max importance across regimes
        - per_regime_importances: dict mapping regime_id -> pd.Series of importances
    """
    # 1. Detect regimes
    regimes = _detect_regimes(df, n_regimes)
    if regimes is None:
        logger.warning("Regime detection failed, skipping regime-conditional selection")
        return None

    # 2. Compute importance per regime
    per_regime: dict[int, pd.Series] = {}
    unique_regimes = sorted(regimes.dropna().unique().astype(int))

    for regime_id in unique_regimes:
        regime_mask = regimes == regime_id
        n_samples = regime_mask.sum()

        if n_samples < min_samples_per_regime:
            logger.info(
                f"    Regime {regime_id}: {n_samples} samples "
                f"(< {min_samples_per_regime}), skipped"
            )
            continue

        # Extract regime subset
        df_regime = df.loc[regime_mask]
        X = df_regime[feature_names].dropna()
        # Align y to X's index after dropna
        y = df_regime.loc[X.index, label_col]

        if len(X) < min_samples_per_regime:
            logger.info(
                f"    Regime {regime_id}: {len(X)} clean rows "
                f"(< {min_samples_per_regime}), skipped"
            )
            continue

        if y.nunique() < 2:
            logger.info(f"    Regime {regime_id}: < 2 label classes, skipped")
            continue

        # Subsample large regime subsets
        max_rows = 30_000
        if len(X) > max_rows:
            from sklearn.model_selection import train_test_split

            X, _, y, _ = train_test_split(
                X, y, train_size=max_rows, stratify=y, random_state=random_state
            )

        # Run lightweight MDA
        try:
            rf = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=5,
                n_jobs=-1,
                random_state=random_state,
            )
            rf.fit(X, y)

            result = permutation_importance(
                rf,
                X,
                y,
                n_repeats=n_repeats,
                random_state=random_state,
                n_jobs=-1,
            )
            importance = pd.Series(result.importances_mean, index=X.columns)
            per_regime[regime_id] = importance

            top3 = importance.nlargest(3)
            logger.info(
                f"    Regime {regime_id}: {len(X)} samples, "
                f"top features: {', '.join(f'{f}={v:.4f}' for f, v in top3.items())}"
            )
        except Exception as e:
            logger.warning(f"    Regime {regime_id}: MDA failed: {e}")
            continue

    if not per_regime:
        logger.warning("No regimes had sufficient data for MDA")
        return None

    # 3. Combined importance = max across regimes (union approach)
    all_features_idx = pd.Index(feature_names)
    combined = pd.Series(0.0, index=all_features_idx)

    for _regime_id, imp in per_regime.items():
        # Reindex to full feature set, fill missing with 0
        imp_full = imp.reindex(all_features_idx, fill_value=0.0)
        combined = combined.clip(lower=0)  # Safety
        combined = pd.DataFrame({"current": combined, "new": imp_full}).max(axis=1)

    combined = combined.sort_values(ascending=False)

    logger.info(
        f"  Regime-conditional selection: {len(per_regime)} regimes analyzed, "
        f"{(combined > 0).sum()} features with positive importance"
    )

    return combined, per_regime


def _detect_regimes(
    df: pd.DataFrame,
    n_regimes: int = 2,
) -> pd.Series | None:
    """Detect volatility regimes from price data.

    Uses rolling volatility ratio (short/long window) to classify bars
    into low-vol and high-vol regimes (or low/medium/high for n_regimes=3).

    Args:
        df: DataFrame with 'close' column.
        n_regimes: Number of regime states (2 or 3).

    Returns:
        pd.Series of regime labels (0, 1, [2]) or None on failure.
    """
    if "close" not in df.columns:
        logger.warning("Regime detection requires 'close' column")
        return None

    try:
        close = df["close"].astype(float)
        log_ret = np.log(close / close.shift(1))

        short_vol = log_ret.rolling(window=20, min_periods=10).std()
        long_vol = log_ret.rolling(window=60, min_periods=30).std()

        vol_ratio = short_vol / long_vol.replace(0, np.nan)
        vol_ratio = vol_ratio.dropna()

        if n_regimes == 2:
            median = vol_ratio.median()
            regimes = (vol_ratio >= median).astype(int)
        else:
            p33 = vol_ratio.quantile(0.333)
            p67 = vol_ratio.quantile(0.667)
            regimes = pd.Series(1, index=vol_ratio.index)  # Medium
            regimes[vol_ratio < p33] = 0  # Low
            regimes[vol_ratio > p67] = 2  # High

        # Reindex to original DataFrame index
        result = pd.Series(np.nan, index=df.index)
        result.loc[regimes.index] = regimes
        return result

    except Exception as e:
        logger.warning(f"Regime detection failed: {e}")
        return None


__all__ = ["compute_regime_importance"]
