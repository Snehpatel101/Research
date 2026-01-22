from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.data.pipeline.stages.labeling.meta import BetSizeMethod, MetaLabeler


class MetaLearner:
    """
    Orchestrates the training and usage of a Meta-Model for trade filtering.

    The Meta-Model learns to predict 'Success' (1) or 'Failure' (0) of the
    Primary Model's signals.
    """

    def __init__(
        self,
        primary_model: Any,
        meta_model_type: str = "random_forest",
        meta_model_params: dict[str, Any] | None = None,
    ):
        self.primary_model = primary_model
        self.meta_model = None
        self.meta_model_type = meta_model_type
        self.meta_model_params = meta_model_params or {
            "n_estimators": 100,
            "max_depth": 5,
            "random_state": 42,
            "class_weight": "balanced",
        }
        self.meta_labeler = None

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        prices_train: pd.Series,  # Needed for meta-labeling returns
        horizon: int,
    ) -> dict[str, float]:
        """
        Train the meta-model.

        Args:
            X_train: Feature matrix
            y_train: Primary labels (multiclass: 0=Short, 1=Hold, 2=Long)
            prices_train: Close prices for calculating outcome
            horizon: Forecast horizon

        Returns:
            Dict of training metrics
        """
        # 1. Generate Primary Predictions (on training data - usually requires OOF,
        #    but for simplicity here we assume X_train is disjoint from what Primary was trained on
        #    OR we accept some overfitting for this demo. Ideally use Cross-Val predictions).
        #    NOTE: In production, use OOF predictions!

        # Check if primary model has predict_proba
        if hasattr(self.primary_model, "predict_proba"):
            primary_probs = self.primary_model.predict_proba(X_train)
            primary_preds = np.argmax(primary_probs, axis=1)
            np.max(primary_probs, axis=1)
        else:
            primary_preds = self.primary_model.predict(X_train)
            np.ones_like(primary_preds, dtype=float)

        # 2. Generate Meta-Labels
        # We need a DataFrame with primary signals and returns
        df_meta = pd.DataFrame(index=X_train.index)
        df_meta["primary_signal"] = 0

        # Map class indices to signals (-1, 0, 1)
        # Assuming mapping: 0->-1 (Short), 1->0 (Hold), 2->1 (Long)
        # Check config for actual mapping, but this is standard in this project
        signal_map = {0: -1, 1: 0, 2: 1}
        df_meta["primary_signal"] = pd.Series(primary_preds).map(signal_map).values
        df_meta["close"] = prices_train.values

        self.meta_labeler = MetaLabeler(
            primary_signal_column="primary_signal",
            bet_size_method=BetSizeMethod.PROBABILITY,
            horizon=horizon,
        )

        # Compute labels
        result = self.meta_labeler.compute_labels(df_meta, horizon=horizon)
        y_meta = result.labels

        # Filter valid samples (exclude -99 and typically exclude Hold signals from primary)
        valid_mask = y_meta != -99

        X_meta = X_train[valid_mask].copy()
        y_meta_valid = y_meta[valid_mask]

        # Add primary confidence as feature?
        # X_meta["primary_confidence"] = confidence[valid_mask]

        # 3. Train Meta-Model
        if self.meta_model_type == "random_forest":
            self.meta_model = RandomForestClassifier(**self.meta_model_params)
            self.meta_model.fit(X_meta, y_meta_valid)

        # 4. Evaluate on Train (just for sanity)
        preds = self.meta_model.predict(X_meta)
        return {
            "accuracy": accuracy_score(y_meta_valid, preds),
            "precision": precision_score(y_meta_valid, preds, zero_division=0),
            "recall": recall_score(y_meta_valid, preds, zero_division=0),
            "f1": f1_score(y_meta_valid, preds, zero_division=0),
        }

    def predict(self, X: pd.DataFrame, primary_preds: np.ndarray) -> np.ndarray:
        """
        Make meta-predictions (should we trade?).

        Args:
            X: Features
            primary_preds: Predictions from primary model (0, 1, 2)

        Returns:
            Binary array (1=Trade, 0=Pass)
        """
        if self.meta_model is None:
            raise ValueError("Meta-model not trained")

        # We might want to filter only non-neutral primary predictions
        # But to return an aligned array, we predict for all, then mask

        meta_preds = self.meta_model.predict(X)

        # Logic: If Meta says 1 (Success) AND Primary says Trade (Short/Long) -> 1
        # If Meta says 0 (Fail) -> 0

        final_decision = np.zeros_like(primary_preds)

        # Indices where primary is Short(0) or Long(2)
        trade_mask = primary_preds != 1

        # Where trade_mask is True, we use meta_preds
        # Note: meta_model was trained only on valid trades.
        # Does it handle the distribution shift?
        # Standard approach: Apply to all, mask later.

        final_decision[trade_mask] = meta_preds[trade_mask]

        return final_decision
