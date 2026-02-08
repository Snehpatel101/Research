"""
Meta-Labeling Pipeline Stage Runner.

This module provides the pipeline integration for meta-labeling, creating
a two-stage approach for trading decisions:

Stage Flow:
1. Train primary classifier on direction labels (optimized for recall)
2. Generate meta-labels (correctness of primary predictions)
3. Train meta-model on meta-labels (predict confidence)
4. Compute bet sizes from meta-model confidence

Integration Points:
- Runs after final_labels stage (requires direction labels)
- Outputs feed into stacking pipeline (meta-labels for ensemble weighting)
- Can be used standalone for bet sizing in production

Engineering Rules Applied:
- Clear contracts with type hints
- Input validation at boundaries
- Fail fast with explicit errors
- Comprehensive logging and artifact tracking
"""

import json
import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from src.data.pipeline.utils import create_failed_result, create_stage_result
from src.validation.cv.purged_kfold import PurgedKFold, PurgedKFoldConfig

from .bet_sizer import BetSizer
from .meta_labeler import MetaLabelGenerator
from .primary_model import PrimaryClassifier

if TYPE_CHECKING:
    from src.core.common.manifest import ArtifactManifest
    from src.data.pipeline.data_config import DataConfig as PipelineConfig
    from src.data.pipeline.utils import StageResult

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# Default recall target for primary model
DEFAULT_RECALL_TARGET = 0.70

# Default minimum confidence for bet sizing
DEFAULT_MIN_CONFIDENCE = 0.55

# Default maximum position size
DEFAULT_MAX_POSITION = 10

# Feature columns to exclude from primary model training
EXCLUDE_FROM_FEATURES = {
    "datetime",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "volume",
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _get_feature_columns(df: pd.DataFrame, horizon: int) -> list[str]:
    """Extract feature columns from DataFrame, excluding metadata and labels."""
    feature_cols = []

    for col in df.columns:
        # Skip metadata columns
        if col in EXCLUDE_FROM_FEATURES:
            continue

        # Skip label columns
        if col.startswith("label_h") or col.startswith("meta_label_h"):
            continue

        # Skip weight columns
        if col.startswith("sample_weight_h"):
            continue

        # Skip other horizon-specific columns
        if any(
            col.startswith(prefix)
            for prefix in [
                "bars_to_hit_h",
                "mae_h",
                "mfe_h",
                "touch_type_h",
                "quality_h",
                "pain_to_gain_h",
                "time_weighted_dd_h",
                "fwd_return_h",
                "fwd_return_log_h",
                "label_end_time_h",
                "correctness_margin_h",
                "primary_pred_h",
                "meta_proba_h",
                "bet_size_h",
            ]
        ):
            continue

        feature_cols.append(col)

    return feature_cols


def _validate_meta_labeling_inputs(
    df: pd.DataFrame,
    horizon: int,
    symbol: str = "",
    timeframe: str = "",
) -> None:
    """
    Validate inputs for meta-labeling stage.

    Args:
        df: DataFrame with labeled data
        horizon: Label horizon
        symbol: Symbol for context
        timeframe: Timeframe for context

    Raises:
        ValueError: If validation fails
    """
    if df.empty:
        raise ValueError(f"Empty DataFrame for {symbol} @ {timeframe}")

    # Check for required label column
    label_col = f"label_h{horizon}"
    if label_col not in df.columns:
        raise ValueError(
            f"Label column '{label_col}' not found for {symbol} @ {timeframe}. "
            f"Ensure final_labels stage completed successfully."
        )

    # Check for required return column (for meta-labeling)
    return_col = f"fwd_return_h{horizon}"
    if return_col not in df.columns:
        logger.warning(
            f"Forward return column '{return_col}' not found. "
            f"Meta-labeling will use direction-only correctness."
        )


def _train_primary_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    sample_weights: np.ndarray | None,
    config: dict[str, Any],
) -> tuple[PrimaryClassifier, dict[str, float]]:
    """
    Train the primary classifier.

    Args:
        X_train: Training features
        y_train: Training labels
        sample_weights: Optional sample weights
        config: Configuration dict

    Returns:
        Tuple of (trained classifier, metrics dict)
    """
    recall_target = config.get("recall_target", DEFAULT_RECALL_TARGET)
    base_model = config.get("base_model", "logistic")

    logger.info(f"Training primary model (target recall={recall_target:.0%}, base={base_model})")

    primary = PrimaryClassifier(
        recall_target=recall_target,
        base_model=base_model,
        calibrate=True,
        cv_folds=5,
        random_state=42,
        class_weight="balanced",
    )

    primary.fit(X_train, y_train, sample_weight=sample_weights)

    metrics = primary.get_recall_metrics()
    return primary, metrics


def _generate_meta_labels(
    primary: PrimaryClassifier,
    X: np.ndarray,
    y_true: np.ndarray,
    returns: np.ndarray | None,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """
    Generate meta-labels using trained primary model.

    Args:
        primary: Trained PrimaryClassifier
        X: Features
        y_true: True direction labels
        returns: Forward returns (optional)
        horizon: Label horizon

    Returns:
        Tuple of (meta_labels, primary_predictions, metrics)
    """
    # Get primary predictions (direction)
    y_primary = primary.predict_side(X, y_direction=y_true)

    # Generate meta-labels
    generator = MetaLabelGenerator(horizon=horizon)
    result = generator.generate(
        y_true=y_true,
        y_primary=y_primary,
        returns=returns,
    )

    return result.meta_labels, y_primary, result.quality_metrics


def _purged_oof_predictions(
    X: np.ndarray,
    y: np.ndarray,
    sample_weights: np.ndarray | None,
    config: dict[str, Any],
    n_splits: int = 5,
    purge_bars: int = 60,
    embargo_bars: int = 200,
) -> tuple[PrimaryClassifier, np.ndarray, np.ndarray]:
    """
    Generate out-of-fold predictions using purged cross-validation.

    Each sample gets predicted only by a model that never saw it during
    training, with purge/embargo gaps to prevent label leakage.

    After OOF predictions, a final model is trained on ALL data for
    downstream use (threshold is set from OOF metrics).

    Args:
        X: Feature matrix (n_samples, n_features)
        y: Labels
        sample_weights: Optional sample weights
        config: Config dict with recall_target, base_model
        n_splits: Number of CV folds
        purge_bars: Bars to purge before test fold
        embargo_bars: Bars to embargo after test fold

    Returns:
        Tuple of (final_trained_model, oof_probabilities, oof_predictions)
        Every sample gets a prediction from a model that never trained on it.
        Purge/embargo only reduces training sets, not test coverage.
    """
    recall_target = config.get("recall_target", DEFAULT_RECALL_TARGET)
    base_model = config.get("base_model", "logistic")

    n_samples = len(y)
    oof_proba = np.full(n_samples, np.nan)
    oof_preds = np.full(n_samples, -1, dtype=int)

    cv_config = PurgedKFoldConfig(
        n_splits=n_splits,
        purge_bars=purge_bars,
        embargo_bars=embargo_bars,
    )
    cv = PurgedKFold(cv_config)

    # PurgedKFold.split() expects a DataFrame (checks .index for DatetimeIndex)
    X_df = pd.DataFrame(X)

    fold_metrics = []
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_df)):
        X_fold_train, y_fold_train = X[train_idx], y[train_idx]
        X_fold_test = X[test_idx]

        fold_weights = sample_weights[train_idx] if sample_weights is not None else None

        fold_model = PrimaryClassifier(
            recall_target=recall_target,
            base_model=base_model,
            calibrate=True,
            cv_folds=min(3, n_splits - 1),
            random_state=42,
            class_weight="balanced",
        )
        fold_model.fit(X_fold_train, y_fold_train, sample_weight=fold_weights)

        fold_proba = fold_model.predict_proba(X_fold_test)[:, 1]
        fold_preds = fold_model.predict(X_fold_test)

        oof_proba[test_idx] = fold_proba
        oof_preds[test_idx] = fold_preds

        fold_metrics.append(fold_model.get_recall_metrics())
        logger.debug(
            f"    Fold {fold_idx}: recall={fold_metrics[-1]['achieved_recall']:.2%}, "
            f"train={len(train_idx)}, test={len(test_idx)}"
        )

    # Train final model on all data (for threshold + downstream predict_side)
    final_model = PrimaryClassifier(
        recall_target=recall_target,
        base_model=base_model,
        calibrate=True,
        cv_folds=min(5, n_splits),
        random_state=42,
        class_weight="balanced",
    )
    final_model.fit(X, y, sample_weight=sample_weights)

    n_predicted = np.sum(~np.isnan(oof_proba))
    logger.info(
        f"    Purged OOF: {n_predicted}/{n_samples} samples predicted "
        f"({n_predicted / n_samples:.1%} coverage)"
    )

    return final_model, oof_proba, oof_preds


def _compute_bet_sizes(
    meta_probs: np.ndarray,
    directions: np.ndarray,
    volatility: np.ndarray | None,
    config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Compute bet sizes from meta-model probabilities.

    Args:
        meta_probs: Meta-model confidence probabilities
        directions: Primary model directions
        volatility: Volatility values (optional)
        config: Configuration dict

    Returns:
        Tuple of (bet_sizes, metadata)
    """
    max_position = config.get("max_position", DEFAULT_MAX_POSITION)
    min_confidence = config.get("min_confidence", DEFAULT_MIN_CONFIDENCE)
    method = config.get("sizing_method", "linear")

    sizer = BetSizer(
        max_position=max_position,
        min_confidence=min_confidence,
        method=method,
        discrete_sizes=True,
    )

    # Use ATR as volatility proxy if available
    vol = None
    if volatility is not None:
        vol = volatility

    result = sizer.compute_sizes(
        meta_probabilities=meta_probs,
        directions=directions,
        volatility=vol,
    )

    return result.position_sizes, result.metadata or {}


# =============================================================================
# MAIN STAGE RUNNER
# =============================================================================


def run_meta_labeling(
    config: "PipelineConfig",
    manifest: "ArtifactManifest",
) -> "StageResult":
    """
    Pipeline Stage: Meta-Labeling for Bet Sizing.

    This stage implements the Lopez de Prado meta-labeling approach:
    1. Train high-recall primary classifier on direction labels
    2. Generate meta-labels (1 if primary correct, 0 if incorrect)
    3. Add meta-labels to dataset for downstream meta-model training

    Args:
        config: Pipeline configuration
        manifest: Artifact manifest for tracking outputs

    Returns:
        StageResult with status and artifacts

    Note:
        This stage should run after final_labels stage.
        Meta-labels are added to the labeled parquet files.
    """
    start_time = datetime.now()
    logger.info("=" * 70)
    logger.info("META-LABELING STAGE: Bet Sizing Labels")
    logger.info("=" * 70)

    try:
        # Get output directories
        meta_label_dir = config.run_data_dir / "meta_labels"
        meta_label_dir.mkdir(parents=True, exist_ok=True)

        # Get effective output timeframes
        output_timeframes = config.effective_output_timeframes
        is_multi_tf = len(output_timeframes) > 1

        logger.info(f"Output timeframes: {output_timeframes}")
        logger.info(f"Multi-TF mode: {is_multi_tf}")

        # Get meta-labeling config from pipeline config
        meta_config = getattr(config, "meta_labeling_config", {}) or {}
        recall_target = meta_config.get("recall_target", DEFAULT_RECALL_TARGET)
        base_model = meta_config.get("base_model", "logistic")
        min_confidence = meta_config.get("min_confidence", DEFAULT_MIN_CONFIDENCE)

        logger.info("Meta-labeling config:")
        logger.info(f"  Recall target: {recall_target:.0%}")
        logger.info(f"  Base model: {base_model}")
        logger.info(f"  Min confidence: {min_confidence:.0%}")

        artifacts = []
        all_metrics: dict[str, dict[str, Any]] = {}

        for symbol in config.symbols:
            all_metrics[symbol] = {}

            for tf in output_timeframes:
                logger.info(f"\n{'='*50}")
                logger.info(f"Processing {symbol} @ {tf}")
                logger.info(f"{'='*50}")

                # Load final labeled data
                labeled_path = config.final_data_dir / f"{symbol}_{tf}_labeled.parquet"
                if not labeled_path.exists():
                    if is_multi_tf:
                        raise FileNotFoundError(
                            f"Labeled file not found: {labeled_path}. "
                            f"Ensure final_labels stage completed."
                        )
                    logger.warning(f"Labeled file not found: {labeled_path}")
                    continue

                df = pd.read_parquet(labeled_path)
                logger.info(f"Loaded {len(df):,} rows from {labeled_path.name}")

                tf_metrics: dict[str, Any] = {"horizons": {}}

                # Process each horizon
                for horizon in config.label_horizons:
                    logger.info(f"\n  Horizon {horizon}:")

                    # Validate inputs
                    _validate_meta_labeling_inputs(df, horizon, symbol, tf)

                    # Get feature columns
                    feature_cols = _get_feature_columns(df, horizon)
                    logger.info(f"    Features: {len(feature_cols)}")

                    # Get label and return columns
                    label_col = f"label_h{horizon}"
                    return_col = f"fwd_return_h{horizon}"
                    weight_col = f"sample_weight_h{horizon}"

                    # Prepare data (exclude invalid labels)
                    valid_mask = df[label_col] != -99
                    df_valid = df[valid_mask].copy()

                    if len(df_valid) < 100:
                        logger.warning(
                            f"    Insufficient valid samples ({len(df_valid)}), skipping"
                        )
                        continue

                    X_valid = df_valid[feature_cols].values
                    y_valid = df_valid[label_col].values

                    # Get sample weights if available
                    weights = None
                    if weight_col in df_valid.columns:
                        weights = df_valid[weight_col].values

                    # Step 1: Purged OOF predictions (prevents data leakage)
                    # Each sample is only predicted by a model that never saw it,
                    # with purge/embargo gaps around each fold boundary.
                    primary, oof_proba, oof_preds = _purged_oof_predictions(
                        X=X_valid,
                        y=y_valid,
                        sample_weights=weights,
                        config={
                            "recall_target": recall_target,
                            "base_model": base_model,
                        },
                        n_splits=5,
                        purge_bars=max(horizon * 3, 60),
                        embargo_bars=min(200, len(df_valid) // 20),
                    )

                    primary_metrics = primary.get_recall_metrics()
                    logger.info(
                        f"    Primary model: recall={primary_metrics['achieved_recall']:.2%}, "
                        f"precision={primary_metrics['achieved_precision']:.2%}"
                    )

                    # Step 2: Generate meta-labels using OOF predictions
                    # Use the final model for predict_side (direction), but
                    # probabilities come from OOF to avoid leakage.
                    y_primary = primary.predict_side(X_valid, y_direction=y_valid)

                    returns_valid = (
                        df_valid[return_col].values if return_col in df.columns else None
                    )

                    generator = MetaLabelGenerator(horizon=horizon)
                    meta_result = generator.generate(
                        y_true=y_valid,
                        y_primary=y_primary,
                        returns=returns_valid,
                    )
                    meta_labels_valid = meta_result.meta_labels
                    meta_metrics = meta_result.quality_metrics

                    logger.info(
                        f"    Meta-labels: accuracy={meta_metrics['primary_accuracy']:.2%}, "
                        f"valid={meta_metrics['n_valid']:.0f}/{meta_metrics['n_total']:.0f}"
                    )

                    # Step 3: Use OOF probabilities as meta-proba (leak-free)
                    # Samples in purge/embargo gaps get NaN; fill with 0.5 (neutral)
                    meta_proba_valid = np.where(np.isnan(oof_proba), 0.5, oof_proba)

                    # Step 4: Compute bet sizes
                    volatility = None
                    if "atr_14" in df_valid.columns:
                        volatility = df_valid["atr_14"].values

                    bet_sizes_valid, sizing_meta = _compute_bet_sizes(
                        meta_probs=meta_proba_valid,
                        directions=y_primary,
                        volatility=volatility,
                        config={
                            "min_confidence": min_confidence,
                            "sizing_method": meta_config.get("sizing_method", "linear"),
                            "max_position": meta_config.get(
                                "max_position", DEFAULT_MAX_POSITION
                            ),
                        },
                    )

                    logger.info(
                        f"    Bet sizes: {sizing_meta['n_active_positions']} active positions"
                    )

                    # Map valid-only results back to full DataFrame
                    # Invalid samples (-99 labels) get neutral defaults
                    n_full = len(df)
                    full_meta_labels = np.full(n_full, 0, dtype=int)
                    full_primary_pred = np.full(n_full, 0, dtype=int)
                    full_meta_proba = np.full(n_full, 0.5)
                    full_bet_sizes = np.zeros(n_full)

                    full_meta_labels[valid_mask] = meta_labels_valid
                    full_primary_pred[valid_mask] = y_primary
                    full_meta_proba[valid_mask] = meta_proba_valid
                    full_bet_sizes[valid_mask] = bet_sizes_valid

                    df[f"meta_label_h{horizon}"] = full_meta_labels
                    df[f"primary_pred_h{horizon}"] = full_primary_pred
                    df[f"meta_proba_h{horizon}"] = full_meta_proba
                    df[f"bet_size_h{horizon}"] = full_bet_sizes

                    # Store metrics
                    tf_metrics["horizons"][horizon] = {
                        "primary_recall": primary_metrics["achieved_recall"],
                        "primary_precision": primary_metrics["achieved_precision"],
                        "primary_threshold": primary_metrics["optimal_threshold"],
                        "meta_accuracy": meta_metrics["primary_accuracy"],
                        "meta_n_valid": meta_metrics["n_valid"],
                        "meta_n_correct": meta_metrics["n_correct"],
                        "active_positions": sizing_meta["n_active_positions"],
                        "oof_coverage": float(
                            np.sum(~np.isnan(oof_proba)) / len(oof_proba)
                        ),
                    }

                # Save updated DataFrame with meta-labels
                output_path = meta_label_dir / f"{symbol}_{tf}_meta_labeled.parquet"
                df.to_parquet(output_path, index=False)
                artifacts.append(output_path)

                manifest.add_artifact(
                    name=f"meta_labeled_{symbol}_{tf}",
                    file_path=output_path,
                    stage="meta_labeling",
                    metadata={
                        "symbol": symbol,
                        "timeframe": tf,
                        "horizons": config.label_horizons,
                    },
                )

                logger.info(f"\n  Saved meta-labeled data to {output_path}")
                all_metrics[symbol][tf] = tf_metrics

        # Save meta-labeling manifest
        manifest_path = meta_label_dir / "meta_labeling_manifest.json"
        meta_manifest = {
            "run_id": config.run_id,
            "created_at": datetime.now().isoformat(),
            "stage": "meta_labeling",
            "config": {
                "recall_target": recall_target,
                "base_model": base_model,
                "min_confidence": min_confidence,
            },
            "symbols": config.symbols,
            "timeframes": output_timeframes,
            "horizons": config.label_horizons,
            "metrics": all_metrics,
            "columns_added": {
                "meta_label": "meta_label_h{h}",
                "primary_pred": "primary_pred_h{h}",
                "meta_proba": "meta_proba_h{h}",
                "bet_size": "bet_size_h{h}",
            },
        }

        with open(manifest_path, "w") as f:
            json.dump(meta_manifest, f, indent=2)
        artifacts.append(manifest_path)

        logger.info(f"\nMeta-labeling manifest saved to {manifest_path}")

        return create_stage_result(
            stage_name="meta_labeling",
            start_time=start_time,
            artifacts=artifacts,
            metadata={
                "symbols": config.symbols,
                "timeframes": output_timeframes,
                "horizons": config.label_horizons,
                "metrics": all_metrics,
            },
        )

    except Exception as e:
        logger.error(f"Meta-labeling failed: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return create_failed_result(
            stage_name="meta_labeling",
            start_time=start_time,
            error=str(e),
        )


# =============================================================================
# STANDALONE UTILITIES
# =============================================================================


def add_meta_labels_standalone(
    df: pd.DataFrame,
    horizon: int,
    recall_target: float = DEFAULT_RECALL_TARGET,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
) -> pd.DataFrame:
    """
    Standalone function to add meta-labels to a DataFrame.

    Uses purged cross-validation to prevent data leakage: the primary model
    never sees the samples it generates predictions for.

    Args:
        df: DataFrame with labels and features
        horizon: Label horizon
        recall_target: Primary model recall target
        min_confidence: Minimum confidence for bet sizing

    Returns:
        DataFrame with meta-label columns added
    """
    logger.info(f"Adding meta-labels for horizon {horizon}")

    # Get feature columns
    feature_cols = _get_feature_columns(df, horizon)
    label_col = f"label_h{horizon}"
    return_col = f"fwd_return_h{horizon}"
    weight_col = f"sample_weight_h{horizon}"

    # Prepare data
    valid_mask = df[label_col] != -99
    df_valid = df[valid_mask].copy()

    X_valid = df_valid[feature_cols].values
    y_valid = df_valid[label_col].values
    weights = df_valid[weight_col].values if weight_col in df_valid.columns else None

    # Purged OOF predictions (leak-free)
    primary, oof_proba, _ = _purged_oof_predictions(
        X=X_valid,
        y=y_valid,
        sample_weights=weights,
        config={"recall_target": recall_target},
        purge_bars=max(horizon * 3, 60),
        embargo_bars=min(200, len(df_valid) // 20),
    )

    # Generate meta-labels
    y_primary = primary.predict_side(X_valid, y_direction=y_valid)
    returns_valid = df_valid[return_col].values if return_col in df.columns else None

    generator = MetaLabelGenerator(horizon=horizon)
    meta_result = generator.generate(
        y_true=y_valid,
        y_primary=y_primary,
        returns=returns_valid,
    )

    # Use OOF probabilities (NaN for purged samples -> fill with 0.5)
    meta_proba_valid = np.where(np.isnan(oof_proba), 0.5, oof_proba)

    volatility = df_valid["atr_14"].values if "atr_14" in df_valid.columns else None
    bet_sizes_valid, _ = _compute_bet_sizes(
        meta_probs=meta_proba_valid,
        directions=y_primary,
        volatility=volatility,
        config={"min_confidence": min_confidence},
    )

    # Map back to full DataFrame
    df_out = df.copy()
    df_out[f"meta_label_h{horizon}"] = 0
    df_out[f"primary_pred_h{horizon}"] = 0
    df_out[f"meta_proba_h{horizon}"] = 0.5
    df_out[f"bet_size_h{horizon}"] = 0.0

    df_out.loc[valid_mask, f"meta_label_h{horizon}"] = meta_result.meta_labels
    df_out.loc[valid_mask, f"primary_pred_h{horizon}"] = y_primary
    df_out.loc[valid_mask, f"meta_proba_h{horizon}"] = meta_proba_valid
    df_out.loc[valid_mask, f"bet_size_h{horizon}"] = bet_sizes_valid

    return df_out
