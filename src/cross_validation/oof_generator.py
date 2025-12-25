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
from src.models.registry import ModelRegistry
from src.models.base import PredictionOutput

logger = logging.getLogger(__name__)


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
    ) -> Dict[str, OOFPrediction]:
        """
        Generate OOF predictions for all models.

        Args:
            X: Feature DataFrame
            y: Labels
            model_configs: Dict mapping model_name to hyperparameters
            sample_weights: Optional quality weights
            feature_subset: Optional subset of features to use

        Returns:
            Dict mapping model_name to OOFPrediction
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
            )
            oof_results[model_name] = oof_pred

            logger.info(
                f"  {model_name}: {oof_pred.predictions.shape[0]} predictions, "
                f"coverage={oof_pred.coverage:.2%}"
            )

        return oof_results

    def _generate_single_model_oof(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_name: str,
        config: Dict[str, Any],
        sample_weights: Optional[pd.Series] = None,
    ) -> OOFPrediction:
        """Generate OOF predictions for a single model."""
        n_samples = len(X)
        n_classes = 3  # short, neutral, long

        # Initialize OOF storage
        oof_probs = np.full((n_samples, n_classes), np.nan)
        oof_preds = np.full(n_samples, np.nan)
        oof_confidence = np.full(n_samples, np.nan)
        fold_info: List[Dict[str, Any]] = []

        # Generate predictions fold by fold
        for fold_idx, (train_idx, val_idx) in enumerate(self.cv.split(X, y)):
            logger.debug(f"  Fold {fold_idx + 1}: train={len(train_idx)}, val={len(val_idx)}")

            # Extract fold data
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Handle sample weights
            if sample_weights is not None:
                w_train = sample_weights.iloc[train_idx].values
                w_val = sample_weights.iloc[val_idx].values
            else:
                w_train = None
                w_val = None

            # Create and train model
            model = ModelRegistry.create(model_name, config=config)

            # Use model's fit interface (expects numpy arrays for most models)
            training_metrics = model.fit(
                X_train=X_train.values,
                y_train=y_train.values,
                X_val=X_val.values,
                y_val=y_val.values,
                sample_weights=w_train,
            )

            # Generate predictions for validation fold
            prediction_output: PredictionOutput = model.predict(X_val.values)

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
