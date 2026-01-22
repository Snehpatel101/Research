"""
Triple-Barrier Labeling Pipeline Stage.

IMPORTANT: This is a THIN WRAPPER around src.labeling.triple_barrier.
Do NOT duplicate implementation logic here.

The authoritative implementation lives in src/labeling/triple_barrier.py which contains:
- Numba-optimized triple barrier computation
- TripleBarrierConfig for configuration
- TripleBarrierLabeler for the labeling strategy
- Transaction cost adjustment
- Adaptive barrier scaling

This wrapper adapts the interface for pipeline stage usage, providing:
- Pipeline-specific configuration defaults via BARRIER_PARAMS_DEFAULT
- Compatibility with the pipeline stage interface
"""

import logging
from typing import Any

import numpy as np
import pandas as pd

# Import from authoritative implementation
from src.labeling.triple_barrier import (
    TripleBarrierConfig,
    TripleBarrierLabeler as AuthoritativeTripleBarrierLabeler,
    triple_barrier_numba,
    triple_barrier_numba_with_costs,
)

# Import base classes from local pipeline base for interface compatibility
from .base import LabelingResult, LabelingStrategy, LabelingType

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class TripleBarrierLabeler(LabelingStrategy):
    """
    Triple-barrier labeling strategy for pipeline usage.

    This is a thin wrapper around src.labeling.triple_barrier.TripleBarrierLabeler
    that provides pipeline-specific defaults and configuration.

    The actual labeling algorithm is delegated to the authoritative implementation
    to ensure consistency across the codebase.

    Parameters
    ----------
    k_up : float, optional
        Multiplier for upper barrier (default from config)
    k_down : float, optional
        Multiplier for lower barrier (default from config)
    max_bars : int, optional
        Maximum bars before timeout (default from config)
    atr_column : str
        Column name for ATR values (default: 'atr_14')
    apply_transaction_costs : bool
        Whether to adjust upper barrier for transaction costs (default: True)
    symbol : str, optional
        Symbol for cost lookup (default: 'MES')
    volatility_regime : str
        Volatility regime for slippage: 'low_vol' or 'high_vol' (default: 'low_vol')

    Example
    -------
    >>> labeler = TripleBarrierLabeler(k_up=2.0, k_down=1.5)
    >>> result = labeler.compute_labels(df, horizon=20)
    >>> print(result.quality_metrics)
    """

    def __init__(
        self,
        k_up: float | None = None,
        k_down: float | None = None,
        max_bars: int | None = None,
        atr_column: str = "atr_14",
        apply_transaction_costs: bool = True,
        symbol: str = "MES",
        volatility_regime: str = "low_vol",
    ):
        self._k_up = k_up
        self._k_down = k_down
        self._max_bars = max_bars
        self._atr_column = atr_column
        self._apply_transaction_costs = apply_transaction_costs
        self._symbol = symbol
        self._volatility_regime = volatility_regime

    @property
    def labeling_type(self) -> LabelingType:
        """Return the type of this labeling strategy."""
        return LabelingType.TRIPLE_BARRIER

    @property
    def required_columns(self) -> list[str]:
        """Return list of required DataFrame columns."""
        return ["close", "high", "low", "open", self._atr_column]

    def _get_default_params(self, horizon: int) -> dict[str, Any]:
        """Get default parameters for a given horizon from pipeline config."""
        try:
            from src.pipeline.config import BARRIER_PARAMS_DEFAULT

            if horizon in BARRIER_PARAMS_DEFAULT:
                return BARRIER_PARAMS_DEFAULT[horizon]
        except ImportError:
            logger.debug("BARRIER_PARAMS_DEFAULT not available, using fallback defaults")

        return {"k_up": 1.0, "k_down": 1.0, "max_bars": max(horizon * 3, 10)}

    def _create_authoritative_config(
        self,
        horizon: int,
        k_up: float | None = None,
        k_down: float | None = None,
        max_bars: int | None = None,
    ) -> TripleBarrierConfig:
        """
        Create a TripleBarrierConfig for the authoritative labeler.

        Merges instance defaults, horizon defaults, and method overrides.
        """
        defaults = self._get_default_params(horizon)

        return TripleBarrierConfig(
            upper_mult=k_up or self._k_up or defaults["k_up"],
            lower_mult=k_down or self._k_down or defaults["k_down"],
            horizon=max_bars or self._max_bars or defaults["max_bars"],
            atr_column=self._atr_column,
            apply_transaction_costs=self._apply_transaction_costs,
            symbol=self._symbol,
            volatility_regime=self._volatility_regime,
        )

    def compute_labels(
        self,
        df: pd.DataFrame,
        horizon: int,
        k_up: float | None = None,
        k_down: float | None = None,
        max_bars: int | None = None,
        apply_transaction_costs: bool | None = None,
        symbol: str | None = None,
        volatility_regime: str | None = None,
        **kwargs: Any,
    ) -> LabelingResult:
        """
        Compute triple-barrier labels for the given horizon.

        Delegates to the authoritative implementation in src.labeling.triple_barrier.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with OHLCV data and ATR
        horizon : int
            Horizon identifier (e.g., 1, 5, 20)
        k_up : float, optional
            Override for upper barrier multiplier
        k_down : float, optional
            Override for lower barrier multiplier
        max_bars : int, optional
            Override for maximum bars
        apply_transaction_costs : bool, optional
            Override for transaction cost adjustment
        symbol : str, optional
            Override for symbol
        volatility_regime : str, optional
            Override for volatility regime
        **kwargs : Any
            Additional parameters passed to authoritative implementation

        Returns
        -------
        LabelingResult
            Container with labels and quality metrics
        """
        # Validate inputs using base class method
        self.validate_inputs(df)

        if not isinstance(horizon, int) or horizon <= 0:
            raise ValueError(f"horizon must be a positive integer, got {horizon}")

        # Resolve transaction cost settings (allow method overrides)
        effective_apply_costs = (
            apply_transaction_costs
            if apply_transaction_costs is not None
            else self._apply_transaction_costs
        )
        effective_symbol = symbol or self._symbol
        effective_volatility_regime = volatility_regime or self._volatility_regime

        # Create config for authoritative labeler
        config = TripleBarrierConfig(
            upper_mult=k_up or self._k_up or self._get_default_params(horizon)["k_up"],
            lower_mult=k_down or self._k_down or self._get_default_params(horizon)["k_down"],
            horizon=max_bars or self._max_bars or self._get_default_params(horizon)["max_bars"],
            atr_column=self._atr_column,
            apply_transaction_costs=effective_apply_costs,
            symbol=effective_symbol,
            volatility_regime=effective_volatility_regime,
        )

        # Create authoritative labeler and delegate
        authoritative_labeler = AuthoritativeTripleBarrierLabeler(config)

        # Delegate to authoritative implementation
        # The authoritative compute_labels returns a LabelingResult from src.labeling.base
        # which is compatible with our local LabelingResult
        auth_result = authoritative_labeler.compute_labels(
            df=df,
            horizon=horizon,
            k_up=config.upper_mult,
            k_down=config.lower_mult,
            max_bars=config.horizon,
            **kwargs,
        )

        # Convert to local LabelingResult for interface compatibility
        # Both LabelingResult classes have the same structure
        result = LabelingResult(
            labels=auth_result.labels,
            horizon=auth_result.horizon,
            metadata=auth_result.metadata,
            quality_metrics=auth_result.quality_metrics,
        )

        return result

    def get_quality_metrics(self, result: LabelingResult) -> dict[str, float]:
        """
        Compute quality metrics specific to triple-barrier labeling.

        Delegates MAE/MFE analysis to base implementation and adds
        triple-barrier specific metrics.
        """
        metrics = super().get_quality_metrics(result)

        # Add triple-barrier specific metrics from metadata
        labels = result.labels
        valid_mask = labels != -99

        mae = result.metadata.get("mae", np.array([]))
        mfe = result.metadata.get("mfe", np.array([]))
        bars_to_hit = result.metadata.get("bars_to_hit", np.array([]))

        if len(mae) > 0 and len(mfe) > 0:
            valid_mae = mae[valid_mask]
            valid_mfe = mfe[valid_mask]

            if len(valid_mae) > 0:
                metrics["avg_mae"] = float(np.mean(valid_mae))
                metrics["avg_mfe"] = float(np.mean(valid_mfe))

                # Risk-reward ratio (MFE/|MAE|)
                avg_mae_abs = abs(metrics["avg_mae"]) if metrics["avg_mae"] != 0 else 1e-6
                metrics["mfe_mae_ratio"] = abs(metrics["avg_mfe"]) / avg_mae_abs

        if len(bars_to_hit) > 0:
            valid_bars = bars_to_hit[valid_mask]
            if len(valid_bars) > 0 and valid_bars.sum() > 0:
                non_zero_bars = valid_bars[valid_bars > 0]
                if len(non_zero_bars) > 0:
                    metrics["avg_bars_to_hit"] = float(np.mean(non_zero_bars))

        return metrics


# Re-export Numba functions for backward compatibility
# These are imported from the authoritative implementation
__all__ = [
    "TripleBarrierLabeler",
    "triple_barrier_numba",
    "triple_barrier_numba_with_costs",
]
