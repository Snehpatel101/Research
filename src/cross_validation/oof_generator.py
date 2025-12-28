"""
Out-of-Fold (OOF) Prediction Generator.

Generates truly out-of-sample predictions where each sample is predicted
by a model that never saw that sample during training. These OOF predictions
become training data for Phase 4 ensemble stacking.

Why OOF predictions matter:
- In-sample predictions are overconfident (overfitting)
- OOF predictions reflect realistic model performance
- Meta-learner trains on honest prediction quality
- Better generalization to new data
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.cross_validation.purged_kfold import PurgedKFold
from src.cross_validation.fold_scaling import FoldAwareScaler, get_scaling_method_for_model
from src.cross_validation.sequence_cv import SequenceCVBuilder, SequenceFoldResult
from src.models.registry import ModelRegistry
from src.models.base import PredictionOutput
from src.models.calibration import CalibrationConfig, ProbabilityCalibrator

logger = logging.getLogger(__name__)

# Default sequence length for sequence models
DEFAULT_SEQUENCE_LENGTH = 60


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class OOFPrediction:
    """
    Out-of-fold predictions for a single model.

    Attributes:
        model_name: Name of the model
        predictions: DataFrame with OOF predictions
        fold_info: Per-fold training information
        coverage: Fraction of samples with predictions
    """
    model_name: str
    predictions: pd.DataFrame
    fold_info: List[Dict[str, Any]]
    coverage: float = 1.0

    def get_probabilities(self) -> np.ndarray:
        """Get probability matrix (n_samples, 3)."""
        return self.predictions[
            [f"{self.model_name}_prob_short",
             f"{self.model_name}_prob_neutral",
             f"{self.model_name}_prob_long"]
        ].values

    def get_class_predictions(self) -> np.ndarray:
        """Get predicted classes (-1, 0, 1)."""
        return self.predictions[f"{self.model_name}_pred"].values


@dataclass
class StackingDataset:
    """
    Dataset for training ensemble meta-learner.

    Contains OOF predictions from all base models plus true labels.

    Attributes:
        data: DataFrame with all model predictions and derived features
        model_names: List of base model names
        horizon: Label horizon
        metadata: Additional metadata
    """
    data: pd.DataFrame
    model_names: List[str]
    horizon: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_samples(self) -> int:
        return len(self.data)

    @property
    def n_models(self) -> int:
        return len(self.model_names)

    def get_features(self) -> pd.DataFrame:
        """Get feature columns for meta-learner."""
        # Exclude y_true and datetime columns
        feature_cols = [c for c in self.data.columns if c not in ("y_true", "datetime")]
        return self.data[feature_cols]

    def get_labels(self) -> pd.Series:
        """Get true labels."""
        return self.data["y_true"]


# =============================================================================
# OOF GENERATOR
# =============================================================================

class OOFGenerator:
    """
    Generate out-of-fold predictions for stacking.

    Each sample gets a prediction from a model trained without
    seeing that sample. This prevents overfitting in the meta-learner.

    Example:
        >>> oof_gen = OOFGenerator(cv)
        >>> model_configs = {"xgboost": {"max_depth": 6}}
        >>> oof_predictions = oof_gen.generate_oof_predictions(X, y, model_configs)
        >>> stacking_ds = oof_gen.build_stacking_dataset(oof_predictions, y, horizon=20)
    """

    def __init__(self, cv: PurgedKFold) -> None:
        """
        Initialize OOFGenerator.

        Args:
            cv: PurgedKFold cross-validator
        """
        self.cv = cv

    def generate_oof_predictions(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_configs: Dict[str, Dict[str, Any]],
        sample_weights: Optional[pd.Series] = None,
        feature_subset: Optional[List[str]] = None,
        calibrate: bool = False,
        calibration_method: str = "auto",
        label_end_times: Optional[pd.Series] = None,
    ) -> Dict[str, OOFPrediction]:
        """
        Generate OOF predictions for all models.

        Args:
            X: Feature DataFrame
            y: Labels
            model_configs: Dict mapping model_name to hyperparameters
            sample_weights: Optional quality weights
            feature_subset: Optional subset of features to use
            calibrate: Whether to apply probability calibration to OOF predictions
            calibration_method: Calibration method ("auto", "isotonic", "sigmoid")
            label_end_times: Optional Series of datetime when each label is resolved.
                If provided, enables proper purging of overlapping labels in CV.

        Returns:
            Dict mapping model_name to OOFPrediction

        Note:
            Calibration is leakage-safe because OOF predictions are already
            out-of-sample (each prediction is from a model that never saw
            that sample). The calibrator learns the mapping between OOF
            probability outputs and actual outcomes.
        """
        oof_results: Dict[str, OOFPrediction] = {}

        # Apply feature subset if specified
        if feature_subset:
            X = X[feature_subset]

        for model_name, config in model_configs.items():
            logger.info(f"Generating OOF predictions for {model_name}...")

            oof_pred = self._generate_single_model_oof(
                X=X,
                y=y,
                model_name=model_name,
                config=config,
                sample_weights=sample_weights,
                label_end_times=label_end_times,
            )
            oof_results[model_name] = oof_pred

            logger.info(
                f"  {model_name}: {oof_pred.predictions.shape[0]} predictions, "
                f"coverage={oof_pred.coverage:.2%}"
            )

        # Apply calibration if requested (leakage-safe: OOF predictions are out-of-sample)
        if calibrate:
            oof_results = self._calibrate_oof_predictions(
                oof_results, y, calibration_method
            )

        return oof_results

    def _calibrate_oof_predictions(
        self,
        oof_results: Dict[str, OOFPrediction],
        y_true: pd.Series,
        calibration_method: str = "auto",
    ) -> Dict[str, OOFPrediction]:
        """
        Apply probability calibration to OOF predictions.

        This is leakage-safe because OOF predictions are truly out-of-sample:
        each prediction was made by a model that never saw that sample during
        training. The calibrator learns the probability mapping from these
        honest predictions.

        Args:
            oof_results: Dict of OOF predictions by model
            y_true: True labels
            calibration_method: Calibration method

        Returns:
            Dict of calibrated OOF predictions
        """
        logger.info("Applying probability calibration to OOF predictions...")

        y_array = y_true.values

        for model_name, oof_pred in oof_results.items():
            # Get probability columns
            prob_cols = [
                f"{model_name}_prob_short",
                f"{model_name}_prob_neutral",
                f"{model_name}_prob_long",
            ]
            probs = oof_pred.predictions[prob_cols].values

            # Handle NaN predictions (keep them as-is)
            valid_mask = ~np.isnan(probs[:, 0])
            if valid_mask.sum() == 0:
                logger.warning(f"  {model_name}: No valid predictions to calibrate")
                continue

            valid_probs = probs[valid_mask]
            valid_y = y_array[valid_mask]

            # Fit and apply calibrator
            cal_config = CalibrationConfig(method=calibration_method)
            calibrator = ProbabilityCalibrator(cal_config)
            metrics = calibrator.fit(valid_y, valid_probs)

            calibrated_probs = calibrator.calibrate(valid_probs)

            # Update predictions DataFrame
            oof_pred.predictions.loc[valid_mask, prob_cols] = calibrated_probs

            # Update confidence based on calibrated probabilities
            oof_pred.predictions.loc[valid_mask, f"{model_name}_confidence"] = (
                calibrated_probs.max(axis=1)
            )

            logger.info(
                f"  {model_name}: Brier {metrics.brier_before:.4f} -> {metrics.brier_after:.4f}, "
                f"ECE {metrics.ece_before:.4f} -> {metrics.ece_after:.4f}"
            )

        return oof_results

    def _generate_single_model_oof(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_name: str,
        config: Dict[str, Any],
        sample_weights: Optional[pd.Series] = None,
        label_end_times: Optional[pd.Series] = None,
    ) -> OOFPrediction:
        """Generate OOF predictions for a single model."""
        n_samples = len(X)
        n_classes = 3  # short, neutral, long

        # Initialize OOF storage
        oof_probs = np.full((n_samples, n_classes), np.nan)
        oof_preds = np.full(n_samples, np.nan)
        oof_confidence = np.full(n_samples, np.nan)
        fold_info: List[Dict[str, Any]] = []

        # Determine scaling method based on model requirements
        scaling_method = get_scaling_method_for_model(model_name)
        fold_scaler = FoldAwareScaler(method=scaling_method)

        # Check if model requires sequences
        try:
            model_info = ModelRegistry.get_model_info(model_name)
            requires_sequences = model_info.get("requires_sequences", False)
        except ValueError:
            requires_sequences = False

        # Route to sequence-specific handler if needed
        if requires_sequences:
            seq_len = config.get("sequence_length", DEFAULT_SEQUENCE_LENGTH)
            return self._generate_sequence_model_oof(
                X=X,
                y=y,
                model_name=model_name,
                config=config,
                seq_len=seq_len,
                sample_weights=sample_weights,
                label_end_times=label_end_times,
            )

        # Generate predictions fold by fold (with label_end_times for overlapping label purge)
        for fold_idx, (train_idx, val_idx) in enumerate(
            self.cv.split(X, y, label_end_times=label_end_times)
        ):
            logger.debug(f"  Fold {fold_idx + 1}: train={len(train_idx)}, val={len(val_idx)}")

            # Extract fold data (raw, unscaled)
            X_train_raw, X_val_raw = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # FOLD-AWARE SCALING: fit scaler on train-only, transform both
            scaling_result = fold_scaler.fit_transform_fold(
                X_train_raw.values, X_val_raw.values
            )
            X_train_scaled = scaling_result.X_train_scaled
            X_val_scaled = scaling_result.X_val_scaled

            # Handle sample weights
            if sample_weights is not None:
                w_train = sample_weights.iloc[train_idx].values
                w_val = sample_weights.iloc[val_idx].values
            else:
                w_train = None
                w_val = None

            # Create and train model
            model = ModelRegistry.create(model_name, config=config)

            # Use model's fit interface with scaled data
            training_metrics = model.fit(
                X_train=X_train_scaled,
                y_train=y_train.values,
                X_val=X_val_scaled,
                y_val=y_val.values,
                sample_weights=w_train,
            )

            # Generate predictions for validation fold (using scaled data)
            prediction_output: PredictionOutput = model.predict(X_val_scaled)

            # Store OOF predictions
            oof_probs[val_idx] = prediction_output.class_probabilities
            oof_preds[val_idx] = prediction_output.class_predictions
            oof_confidence[val_idx] = prediction_output.confidence

            # Track fold info
            fold_info.append({
                "fold": fold_idx,
                "train_size": len(train_idx),
                "val_size": len(val_idx),
                "val_accuracy": training_metrics.val_accuracy,
                "val_f1": training_metrics.val_f1,
            })

        # Validate coverage
        coverage = float((~np.isnan(oof_preds)).mean())
        if coverage < 1.0:
            logger.warning(
                f"{model_name}: Only {coverage:.2%} coverage. "
                f"{int(np.isnan(oof_preds).sum())} samples missing predictions."
            )

        # Build result DataFrame
        oof_df = pd.DataFrame({
            "datetime": X.index if isinstance(X.index, pd.DatetimeIndex) else range(len(X)),
            f"{model_name}_prob_short": oof_probs[:, 0],
            f"{model_name}_prob_neutral": oof_probs[:, 1],
            f"{model_name}_prob_long": oof_probs[:, 2],
            f"{model_name}_pred": oof_preds,
            f"{model_name}_confidence": oof_confidence,
        })

        return OOFPrediction(
            model_name=model_name,
            predictions=oof_df,
            fold_info=fold_info,
            coverage=coverage,
        )

    def _generate_sequence_model_oof(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_name: str,
        config: Dict[str, Any],
        seq_len: int,
        sample_weights: Optional[pd.Series] = None,
        label_end_times: Optional[pd.Series] = None,
        symbol_column: Optional[str] = "symbol",
    ) -> OOFPrediction:
        """
        Generate OOF predictions for a sequence model (LSTM, GRU, TCN, etc.).

        This method properly handles 3D sequence construction for each CV fold:
        1. Builds sequences from fold indices using SequenceCVBuilder
        2. Respects symbol boundaries (no cross-symbol sequences)
        3. Maps predictions back to original sample indices

        Args:
            X: Feature DataFrame
            y: Label Series
            model_name: Name of the sequence model
            config: Model configuration
            seq_len: Sequence length
            sample_weights: Optional sample weights
            label_end_times: Optional label end times for purging
            symbol_column: Column name for symbol isolation

        Returns:
            OOFPrediction with mapped predictions
        """
        n_samples = len(X)
        n_classes = 3  # short, neutral, long

        # Initialize OOF storage at original sample indices
        oof_probs = np.full((n_samples, n_classes), np.nan)
        oof_preds = np.full(n_samples, np.nan)
        oof_confidence = np.full(n_samples, np.nan)
        fold_info: List[Dict[str, Any]] = []

        # Create sequence builder with symbol awareness
        # Check if symbol column exists
        actual_symbol_col = symbol_column if symbol_column in X.columns else None

        seq_builder = SequenceCVBuilder(
            X=X,
            y=y,
            seq_len=seq_len,
            weights=sample_weights,
            symbol_column=actual_symbol_col,
        )

        # Determine scaling method for sequence model
        scaling_method = get_scaling_method_for_model(model_name)
        fold_scaler = FoldAwareScaler(method=scaling_method)

        logger.info(
            f"Generating sequence OOF for {model_name} (seq_len={seq_len}, "
            f"symbol_isolated={actual_symbol_col is not None})"
        )

        # Generate predictions fold by fold
        for fold_idx, (train_idx, val_idx) in enumerate(
            self.cv.split(X, y, label_end_times=label_end_times)
        ):
            # Build 3D sequences for this fold
            # allow_lookback_outside=True: sequence lookback can include data outside fold
            # but TARGET must be in fold
            train_result = seq_builder.build_fold_sequences(
                train_idx, allow_lookback_outside=True
            )
            val_result = seq_builder.build_fold_sequences(
                val_idx, allow_lookback_outside=True
            )

            if train_result.n_sequences == 0 or val_result.n_sequences == 0:
                logger.warning(
                    f"  Fold {fold_idx + 1}: Skipping - insufficient sequences "
                    f"(train={train_result.n_sequences}, val={val_result.n_sequences})"
                )
                continue

            logger.debug(
                f"  Fold {fold_idx + 1}: train_seq={train_result.n_sequences} "
                f"(from {len(train_idx)}), val_seq={val_result.n_sequences} "
                f"(from {len(val_idx)})"
            )

            # FOLD-AWARE SCALING on the 3D sequences
            # Reshape to 2D for scaling, then back to 3D
            train_shape = train_result.X_sequences.shape  # (n_train, seq_len, features)
            val_shape = val_result.X_sequences.shape  # (n_val, seq_len, features)

            # Flatten: (n_samples * seq_len, features)
            X_train_flat = train_result.X_sequences.reshape(-1, train_shape[2])
            X_val_flat = val_result.X_sequences.reshape(-1, val_shape[2])

            scaling_result = fold_scaler.fit_transform_fold(X_train_flat, X_val_flat)

            # Reshape back to 3D
            X_train_scaled = scaling_result.X_train_scaled.reshape(train_shape)
            X_val_scaled = scaling_result.X_val_scaled.reshape(val_shape)

            # Create and train sequence model
            model = ModelRegistry.create(model_name, config=config)

            training_metrics = model.fit(
                X_train=X_train_scaled,
                y_train=train_result.y,
                X_val=X_val_scaled,
                y_val=val_result.y,
                sample_weights=train_result.weights,
            )

            # Generate predictions for validation sequences
            prediction_output: PredictionOutput = model.predict(X_val_scaled)

            # Map predictions back to original indices
            for seq_idx, original_idx in enumerate(val_result.target_indices):
                oof_probs[original_idx] = prediction_output.class_probabilities[seq_idx]
                oof_preds[original_idx] = prediction_output.class_predictions[seq_idx]
                oof_confidence[original_idx] = prediction_output.confidence[seq_idx]

            # Track fold info
            fold_info.append({
                "fold": fold_idx,
                "train_size": len(train_idx),
                "val_size": len(val_idx),
                "train_sequences": train_result.n_sequences,
                "val_sequences": val_result.n_sequences,
                "val_accuracy": training_metrics.val_accuracy,
                "val_f1": training_metrics.val_f1,
            })

        # Validate coverage (expected to be < 100% for sequence models due to lookback)
        coverage = float((~np.isnan(oof_preds)).mean())
        n_missing = int(np.isnan(oof_preds).sum())

        if coverage < 0.9:
            logger.warning(
                f"{model_name}: Low coverage {coverage:.2%} ({n_missing} missing). "
                f"This is expected for sequence models with seq_len={seq_len}."
            )
        else:
            logger.info(
                f"{model_name}: Coverage {coverage:.2%} ({n_missing} samples without predictions)"
            )

        # Build result DataFrame
        oof_df = pd.DataFrame({
            "datetime": X.index if isinstance(X.index, pd.DatetimeIndex) else range(len(X)),
            f"{model_name}_prob_short": oof_probs[:, 0],
            f"{model_name}_prob_neutral": oof_probs[:, 1],
            f"{model_name}_prob_long": oof_probs[:, 2],
            f"{model_name}_pred": oof_preds,
            f"{model_name}_confidence": oof_confidence,
        })

        return OOFPrediction(
            model_name=model_name,
            predictions=oof_df,
            fold_info=fold_info,
            coverage=coverage,
        )

    def validate_oof_coverage(
        self,
        oof_predictions: Dict[str, OOFPrediction],
        original_index: pd.Index,
    ) -> Dict[str, Any]:
        """
        Validate that OOF predictions cover all samples.

        Args:
            oof_predictions: Dict of OOF predictions by model
            original_index: Original DataFrame index

        Returns:
            Validation result dict with passed status and any issues
        """
        validation = {"passed": True, "issues": [], "coverage": {}}

        for model_name, oof_pred in oof_predictions.items():
            # Check for NaN predictions
            nan_count = oof_pred.predictions[f"{model_name}_pred"].isna().sum()
            coverage = 1.0 - (nan_count / len(original_index))

            validation["coverage"][model_name] = coverage

            if nan_count > 0:
                validation["passed"] = False
                validation["issues"].append({
                    "model": model_name,
                    "missing_samples": int(nan_count),
                    "coverage": coverage,
                })

        return validation

    def build_stacking_dataset(
        self,
        oof_predictions: Dict[str, OOFPrediction],
        y_true: pd.Series,
        horizon: int,
        add_derived_features: bool = True,
    ) -> StackingDataset:
        """
        Build stacking dataset from OOF predictions.

        Creates a DataFrame with:
        - model1_prob_short, model1_prob_neutral, model1_prob_long
        - model2_prob_short, model2_prob_neutral, model2_prob_long
        - Derived features (confidence, agreement, entropy)
        - y_true (label)

        Args:
            oof_predictions: Dict of OOF predictions by model
            y_true: True labels
            horizon: Label horizon (for metadata)
            add_derived_features: Whether to add derived features

        Returns:
            StackingDataset for meta-learner training
        """
        model_names = list(oof_predictions.keys())

        # Start with first model's predictions
        first_model = model_names[0]
        stacking_df = oof_predictions[first_model].predictions.copy()

        # Add other models' predictions
        for model_name in model_names[1:]:
            oof_pred = oof_predictions[model_name].predictions
            # Add all columns except datetime (already present)
            for col in oof_pred.columns:
                if col != "datetime":
                    stacking_df[col] = oof_pred[col]

        # Add true labels
        stacking_df["y_true"] = y_true.values

        # Add derived features for meta-learner
        if add_derived_features:
            stacking_df = self._add_stacking_features(stacking_df, model_names)

        # Compute metadata
        metadata = {
            "horizon": horizon,
            "n_models": len(model_names),
            "model_names": model_names,
            "n_samples": len(stacking_df),
            "coverage": {m: oof_predictions[m].coverage for m in model_names},
        }

        return StackingDataset(
            data=stacking_df,
            model_names=model_names,
            horizon=horizon,
            metadata=metadata,
        )

    def _add_stacking_features(
        self,
        df: pd.DataFrame,
        model_names: List[str],
    ) -> pd.DataFrame:
        """Add derived features for meta-learner."""
        df = df.copy()

        # Model predictions (argmax)
        pred_cols = []
        for model in model_names:
            prob_cols = [
                f"{model}_prob_short",
                f"{model}_prob_neutral",
                f"{model}_prob_long"
            ]
            # Prediction already exists, but ensure it's -1, 0, 1 format
            pred_col = f"{model}_pred"
            pred_cols.append(pred_col)

        # Agreement features
        df["models_agree"] = (df[pred_cols].nunique(axis=1) == 1).astype(int)
        df["agreement_count"] = df[pred_cols].apply(
            lambda x: x.value_counts().max() if len(x.dropna()) > 0 else 0,
            axis=1
        )

        # Average confidence
        conf_cols = [f"{model}_confidence" for model in model_names]
        df["avg_confidence"] = df[conf_cols].mean(axis=1)
        df["min_confidence"] = df[conf_cols].min(axis=1)
        df["max_confidence"] = df[conf_cols].max(axis=1)

        # Prediction entropy (uncertainty) per model
        for model in model_names:
            prob_cols = [
                f"{model}_prob_short",
                f"{model}_prob_neutral",
                f"{model}_prob_long"
            ]
            probs = df[prob_cols].values
            # Entropy: -sum(p * log(p))
            entropy = -np.sum(probs * np.log(probs + 1e-10), axis=1)
            df[f"{model}_entropy"] = entropy

        # Average entropy
        entropy_cols = [f"{model}_entropy" for model in model_names]
        df["avg_entropy"] = df[entropy_cols].mean(axis=1)

        # Disagreement measure (std of predictions across models)
        df["prediction_std"] = df[pred_cols].std(axis=1)

        return df

    def save_stacking_dataset(
        self,
        stacking_ds: StackingDataset,
        output_dir: Path,
    ) -> Path:
        """
        Save stacking dataset to parquet.

        Args:
            stacking_ds: StackingDataset to save
            output_dir: Output directory

        Returns:
            Path to saved parquet file
        """
        import json

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save dataset
        parquet_path = output_dir / f"stacking_dataset_h{stacking_ds.horizon}.parquet"
        stacking_ds.data.to_parquet(parquet_path, index=False)

        # Save metadata
        metadata_path = output_dir / f"stacking_metadata_h{stacking_ds.horizon}.json"
        with open(metadata_path, "w") as f:
            json.dump(stacking_ds.metadata, f, indent=2)

        logger.info(f"Saved stacking dataset to {parquet_path}")
        return parquet_path


# =============================================================================
# UTILITIES
# =============================================================================

def analyze_prediction_correlation(
    stacking_df: pd.DataFrame,
    model_names: List[str],
) -> pd.DataFrame:
    """
    Analyze correlation between model predictions.

    Low correlation = good diversity for ensemble.

    Args:
        stacking_df: Stacking dataset DataFrame
        model_names: List of model names

    Returns:
        DataFrame with correlation analysis
    """
    pred_cols = [f"{model}_pred" for model in model_names]
    pred_df = stacking_df[pred_cols]

    # Compute correlation matrix
    corr_matrix = pred_df.corr()

    # Summarize pairwise correlations
    summary = []
    for i, model_i in enumerate(model_names):
        for j, model_j in enumerate(model_names):
            if i < j:
                corr = corr_matrix.loc[f"{model_i}_pred", f"{model_j}_pred"]
                summary.append({
                    "model_1": model_i,
                    "model_2": model_j,
                    "correlation": corr,
                    "diversity_grade": _grade_diversity(corr),
                })

    return pd.DataFrame(summary)


def _grade_diversity(corr: float) -> str:
    """Grade ensemble diversity based on prediction correlation."""
    if corr < 0.3:
        return "Excellent (highly diverse)"
    elif corr < 0.5:
        return "Good"
    elif corr < 0.7:
        return "Moderate"
    elif corr < 0.85:
        return "Low"
    else:
        return "Poor (models too similar)"


__all__ = [
    "OOFPrediction",
    "StackingDataset",
    "OOFGenerator",
    "analyze_prediction_correlation",
]
