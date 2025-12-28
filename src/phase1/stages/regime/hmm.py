"""
Hidden Markov Model (HMM) regime detection using hmmlearn.

Classifies market regimes using unsupervised Gaussian HMM trained on returns.
Supports regime routing for adaptive model selection.

Usage:
    from src.phase1.stages.regime import HMMRegimeDetector

    detector = HMMRegimeDetector(n_states=3, lookback=252)
    regimes = detector.detect(df)

    # Get state probabilities for routing
    regimes, probs = detector.detect_with_probabilities(df)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False

from .base import RegimeDetector, RegimeType

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# =============================================================================
# HMM REGIME LABELS
# =============================================================================

class HMMRegimeLabel(Enum):
    """HMM regime classification labels based on volatility ordering."""
    LOW_VOLATILITY = "low_vol"
    NORMAL = "normal"
    HIGH_VOLATILITY = "high_vol"
    CRISIS = "crisis"  # For n_states=4


# =============================================================================
# HMM CONFIGURATION
# =============================================================================

@dataclass
class HMMConfig:
    """Configuration for HMM regime detector."""
    n_states: int = 3
    lookback: int = 252  # ~1 trading year
    input_type: str = "returns"  # 'returns' or 'volatility'
    covariance_type: str = "full"  # 'spherical', 'diag', 'full', 'tied'
    max_iter: int = 100
    n_init: int = 10  # Number of random initializations
    random_state: int = 42
    min_observations: int = 50  # Minimum obs per state

    def __post_init__(self) -> None:
        if self.n_states < 2:
            raise ValueError(f"n_states must be >= 2, got {self.n_states}")
        if self.lookback < self.n_states * 20:
            raise ValueError(
                f"lookback ({self.lookback}) must be >= {self.n_states * 20} "
                f"for {self.n_states} states"
            )
        if self.input_type not in ("returns", "volatility"):
            raise ValueError(f"input_type must be 'returns' or 'volatility'")
        if self.covariance_type not in ("spherical", "diag", "full", "tied"):
            raise ValueError(f"Invalid covariance_type: {self.covariance_type}")


# =============================================================================
# PURE FUNCTIONS
# =============================================================================

def fit_gaussian_hmm(
    observations: np.ndarray,
    n_states: int = 3,
    covariance_type: str = "full",
    max_iter: int = 100,
    n_init: int = 10,
    random_state: int = 42,
) -> Tuple[Any, np.ndarray, np.ndarray]:
    """
    Fit Gaussian HMM and return model with hidden states.

    Args:
        observations: 1D or 2D array of observations
        n_states: Number of hidden states
        covariance_type: Type of covariance parameters
        max_iter: Maximum EM iterations
        n_init: Number of random initializations
        random_state: Random seed

    Returns:
        Tuple of (fitted_model, hidden_states, state_probabilities)

    Raises:
        ImportError: If hmmlearn is not installed
        ValueError: If observations are invalid
    """
    if not HMM_AVAILABLE:
        raise ImportError(
            "hmmlearn is required for HMM regime detection. "
            "Install with: pip install hmmlearn"
        )

    # Reshape to 2D if needed
    if observations.ndim == 1:
        observations = observations.reshape(-1, 1)

    # Remove NaN
    valid_mask = ~np.isnan(observations).any(axis=1)
    clean_obs = observations[valid_mask]

    if len(clean_obs) < n_states * 10:
        raise ValueError(
            f"Insufficient observations: {len(clean_obs)} for {n_states} states"
        )

    # Fit HMM with multiple initializations
    best_model = None
    best_score = -np.inf

    for init in range(n_init):
        try:
            model = hmm.GaussianHMM(
                n_components=n_states,
                covariance_type=covariance_type,
                n_iter=max_iter,
                random_state=random_state + init,
            )
            model.fit(clean_obs)
            score = model.score(clean_obs)

            if score > best_score:
                best_score = score
                best_model = model
        except Exception as e:
            logger.debug(f"HMM init {init} failed: {e}")
            continue

    if best_model is None:
        raise ValueError("All HMM initializations failed")

    # Predict states for all observations (including NaN positions)
    states = np.full(len(observations), -1, dtype=np.int32)
    probs = np.full((len(observations), n_states), np.nan)

    states[valid_mask] = best_model.predict(clean_obs)
    probs[valid_mask] = best_model.predict_proba(clean_obs)

    return best_model, states, probs


def order_states_by_volatility(
    model: Any,
    current_states: np.ndarray,
) -> Tuple[np.ndarray, Dict[int, int]]:
    """
    Reorder HMM states by increasing volatility for interpretability.

    Args:
        model: Fitted GaussianHMM model
        current_states: Original state labels

    Returns:
        Tuple of (reordered_states, state_mapping)
    """
    # Get state variances (diagonal of covariance)
    if model.covariance_type == "spherical":
        variances = model.covars_
    elif model.covariance_type == "diag":
        variances = model.covars_.mean(axis=1)
    elif model.covariance_type == "full":
        variances = np.array([np.trace(c) for c in model.covars_])
    else:  # tied
        variances = np.array([np.trace(model.covars_)] * model.n_components)

    # Sort by volatility
    order = np.argsort(variances)
    mapping = {old: new for new, old in enumerate(order)}

    # Apply mapping
    new_states = np.array([
        mapping.get(s, s) if s >= 0 else s
        for s in current_states
    ])

    return new_states, mapping


# =============================================================================
# HMM REGIME DETECTOR
# =============================================================================

class HMMRegimeDetector(RegimeDetector):
    """
    Detect market regimes using Hidden Markov Model.

    Uses unsupervised Gaussian HMM to identify latent market states
    based on return dynamics. States are ordered by volatility for
    interpretability (state 0 = lowest vol, state N-1 = highest vol).

    Features:
    - Gaussian emission for continuous returns
    - Multiple random initializations for robustness
    - State ordering by volatility
    - Rolling or expanding window fitting
    - State probability output for regime routing

    Example:
        >>> detector = HMMRegimeDetector(n_states=3, lookback=252)
        >>> regimes = detector.detect(df)
        >>> regimes.value_counts()
        low_vol     0.45
        normal      0.35
        high_vol    0.20
    """

    def __init__(
        self,
        n_states: int = 3,
        lookback: int = 252,
        input_type: str = "returns",
        covariance_type: str = "full",
        max_iter: int = 100,
        n_init: int = 10,
        random_state: int = 42,
        expanding: bool = True,
    ) -> None:
        """
        Initialize HMM regime detector.

        Args:
            n_states: Number of hidden states (2-4 recommended)
            lookback: Minimum observations for fitting
            input_type: 'returns' for log returns, 'volatility' for realized vol
            covariance_type: HMM covariance type
            max_iter: Maximum EM iterations
            n_init: Number of random initializations
            random_state: Random seed
            expanding: If True, use expanding window; else rolling
        """
        super().__init__(RegimeType.COMPOSITE)  # Use COMPOSITE until HMM added

        self.config = HMMConfig(
            n_states=n_states,
            lookback=lookback,
            input_type=input_type,
            covariance_type=covariance_type,
            max_iter=max_iter,
            n_init=n_init,
            random_state=random_state,
        )
        self.expanding = expanding
        self._fitted_model = None
        self._state_mapping = None

    def get_required_columns(self) -> List[str]:
        """Required columns for detection."""
        return ["close"]

    def detect(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect HMM regimes for each bar.

        Args:
            df: DataFrame with 'close' column

        Returns:
            Series with regime labels (categorical)
        """
        regimes, _ = self.detect_with_probabilities(df)
        return regimes

    def detect_with_probabilities(
        self,
        df: pd.DataFrame,
    ) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Detect regimes with state probabilities.

        Args:
            df: DataFrame with 'close' column

        Returns:
            Tuple of (regime_labels, probability_df)
            probability_df has columns: prob_state_0, prob_state_1, ...
        """
        self.validate_input(df)

        if not HMM_AVAILABLE:
            logger.warning("hmmlearn not available, returning NaN regimes")
            regimes = pd.Series(np.nan, index=df.index, dtype="object")
            prob_df = pd.DataFrame(index=df.index)
            return regimes, prob_df

        close = df["close"].values

        # Compute input features
        if self.config.input_type == "returns":
            # Log returns
            observations = np.log(close[1:] / close[:-1])
            observations = np.concatenate([[np.nan], observations])
        else:
            # Rolling realized volatility (20-bar)
            returns = np.log(close[1:] / close[:-1])
            vol = pd.Series(returns).rolling(20).std().values
            observations = np.concatenate([[np.nan] * 20, vol])

        # Initialize output
        n_samples = len(df)
        states = np.full(n_samples, -1, dtype=np.int32)
        probs = np.full((n_samples, self.config.n_states), np.nan)

        # Fit HMM on lookback window
        start_idx = self.config.lookback

        if self.expanding:
            # Fit once on full history, predict for all
            try:
                window_obs = observations[:n_samples]
                self._fitted_model, raw_states, raw_probs = fit_gaussian_hmm(
                    window_obs,
                    n_states=self.config.n_states,
                    covariance_type=self.config.covariance_type,
                    max_iter=self.config.max_iter,
                    n_init=self.config.n_init,
                    random_state=self.config.random_state,
                )

                # Reorder by volatility
                ordered_states, self._state_mapping = order_states_by_volatility(
                    self._fitted_model, raw_states
                )

                states = ordered_states

                # Reorder probability columns to match state reordering
                # state_mapping maps old_state -> new_state
                probs_reordered = np.zeros_like(raw_probs)
                for old_state, new_state in self._state_mapping.items():
                    probs_reordered[:, new_state] = raw_probs[:, old_state]
                probs = probs_reordered

            except Exception as e:
                logger.warning(f"HMM fitting failed: {e}")
        else:
            # Rolling window fitting
            for i in range(start_idx, n_samples):
                window = observations[i - self.config.lookback:i + 1]

                try:
                    model, s, p = fit_gaussian_hmm(
                        window,
                        n_states=self.config.n_states,
                        covariance_type=self.config.covariance_type,
                        max_iter=self.config.max_iter,
                        n_init=self.config.n_init,
                        random_state=self.config.random_state,
                    )

                    ordered_s, _ = order_states_by_volatility(model, s)
                    states[i] = ordered_s[-1]
                    probs[i] = p[-1]

                except Exception as e:
                    logger.debug(f"HMM failed at index {i}: {e}")

        # Convert to labels
        state_labels = self._get_state_labels()
        regimes = pd.Series(
            [state_labels.get(s, np.nan) for s in states],
            index=df.index,
            dtype="object",
        )

        # Build probability DataFrame
        prob_df = pd.DataFrame(
            probs,
            index=df.index,
            columns=[f"prob_state_{i}" for i in range(self.config.n_states)],
        )

        logger.debug(f"HMM regime distribution: {regimes.value_counts(dropna=False).to_dict()}")

        return regimes, prob_df

    def _get_state_labels(self) -> Dict[int, str]:
        """Map state indices to human-readable labels."""
        n = self.config.n_states

        if n == 2:
            return {0: "low_vol", 1: "high_vol"}
        elif n == 3:
            return {0: "low_vol", 1: "normal", 2: "high_vol"}
        elif n == 4:
            return {0: "low_vol", 1: "normal", 2: "high_vol", 3: "crisis"}
        else:
            return {i: f"state_{i}" for i in range(n)}

    def get_transition_matrix(self) -> Optional[np.ndarray]:
        """
        Get the fitted transition probability matrix.

        Returns:
            Transition matrix if fitted, None otherwise
        """
        if self._fitted_model is None:
            return None
        return self._fitted_model.transmat_

    def get_state_means(self) -> Optional[np.ndarray]:
        """
        Get the fitted state means.

        Returns:
            State means if fitted, None otherwise
        """
        if self._fitted_model is None:
            return None
        return self._fitted_model.means_

    def get_state_variances(self) -> Optional[np.ndarray]:
        """
        Get the fitted state variances.

        Returns:
            State variances if fitted, None otherwise
        """
        if self._fitted_model is None:
            return None

        if self._fitted_model.covariance_type == "spherical":
            return self._fitted_model.covars_
        elif self._fitted_model.covariance_type == "diag":
            return self._fitted_model.covars_.mean(axis=1)
        elif self._fitted_model.covariance_type == "full":
            return np.array([np.trace(c) for c in self._fitted_model.covars_])
        else:
            return np.array([np.trace(self._fitted_model.covars_)])


# =============================================================================
# REGIME ROUTER
# =============================================================================

class RegimeRouter:
    """
    Route predictions based on detected regime.

    Selects the appropriate model or strategy based on the current
    market regime. Supports multiple routing strategies.

    Example:
        >>> router = RegimeRouter({
        ...     "low_vol": "trend_model",
        ...     "normal": "balanced_model",
        ...     "high_vol": "volatility_model",
        ... })
        >>> model_name = router.route("high_vol")
        'volatility_model'
    """

    def __init__(
        self,
        regime_model_map: Dict[str, str],
        default_model: Optional[str] = None,
    ) -> None:
        """
        Initialize regime router.

        Args:
            regime_model_map: Mapping of regime labels to model names
            default_model: Model to use when regime is unknown
        """
        self.regime_model_map = regime_model_map
        self.default_model = default_model or list(regime_model_map.values())[0]

    def route(self, regime: str) -> str:
        """
        Get model name for a given regime.

        Args:
            regime: Current regime label

        Returns:
            Model name to use
        """
        return self.regime_model_map.get(regime, self.default_model)

    def route_batch(self, regimes: pd.Series) -> pd.Series:
        """
        Route a batch of regime labels to model names.

        Args:
            regimes: Series of regime labels

        Returns:
            Series of model names
        """
        return regimes.map(lambda r: self.route(r))

    def get_routing_summary(self, regimes: pd.Series) -> Dict[str, Any]:
        """
        Get summary of routing decisions.

        Args:
            regimes: Series of regime labels

        Returns:
            Dict with routing statistics
        """
        routed = self.route_batch(regimes)
        return {
            "model_distribution": routed.value_counts(normalize=True).to_dict(),
            "total_samples": len(regimes),
            "unique_models": routed.nunique(),
        }


__all__ = [
    "HMMRegimeDetector",
    "HMMConfig",
    "HMMRegimeLabel",
    "RegimeRouter",
    "fit_gaussian_hmm",
    "order_states_by_volatility",
]
