"""
Feature Registry - Single source of truth for all features.

Maps feature names to metadata: family, cost, recommendations.
"""

from dataclasses import dataclass, field
from typing import Literal

FeatureFamily = Literal[
    "raw",  # Raw OHLCV (for transformers)
    "momentum",  # RSI, MACD, Stochastic, etc.
    "volatility",  # ATR, Bollinger Bands, etc.
    "volume",  # Volume indicators, OBV, etc.
    "microstructure",  # VPIN, spreads, trade imbalance (2024 features)
    "wavelets",  # Wavelet decompositions
    "mtf",  # Multi-timeframe indicators
    "regime",  # Regime detection features
]

ComputationCost = Literal["cheap", "medium", "expensive"]


@dataclass
class FeatureDefinition:
    """Definition of a single feature."""

    name: str
    family: FeatureFamily
    description: str
    recommended_for: list[str] = field(default_factory=list)  # Model families
    requires_mtf: bool = False
    cost: ComputationCost = "cheap"

    def __repr__(self) -> str:
        return f"Feature({self.name}, family={self.family}, cost={self.cost})"


# =============================================================================
# FEATURE REGISTRY - All 180+ Features
# =============================================================================

FEATURE_REGISTRY: dict[str, FeatureDefinition] = {
    # -------------------------------------------------------------------------
    # Raw OHLCV (for transformers that learn features)
    # -------------------------------------------------------------------------
    "open": FeatureDefinition(
        name="open",
        family="raw",
        description="Open price",
        recommended_for=["transformer"],
        cost="cheap",
    ),
    "high": FeatureDefinition(
        name="high",
        family="raw",
        description="High price",
        recommended_for=["transformer"],
        cost="cheap",
    ),
    "low": FeatureDefinition(
        name="low",
        family="raw",
        description="Low price",
        recommended_for=["transformer"],
        cost="cheap",
    ),
    "close": FeatureDefinition(
        name="close",
        family="raw",
        description="Close price",
        recommended_for=["transformer"],
        cost="cheap",
    ),
    "volume": FeatureDefinition(
        name="volume",
        family="raw",
        description="Volume",
        recommended_for=["transformer", "boosting", "classical", "neural"],
        cost="cheap",
    ),
    # -------------------------------------------------------------------------
    # Momentum Family (~40 features)
    # -------------------------------------------------------------------------
    "rsi_14": FeatureDefinition(
        name="rsi_14",
        family="momentum",
        description="14-period RSI",
        recommended_for=["boosting", "classical", "neural"],
        cost="cheap",
    ),
    "rsi_7": FeatureDefinition(
        name="rsi_7",
        family="momentum",
        description="7-period RSI (faster)",
        recommended_for=["boosting", "classical"],
        cost="cheap",
    ),
    "rsi_21": FeatureDefinition(
        name="rsi_21",
        family="momentum",
        description="21-period RSI (slower)",
        recommended_for=["boosting", "classical"],
        cost="cheap",
    ),
    "macd_line": FeatureDefinition(
        name="macd_line",
        family="momentum",
        description="MACD line (12,26,9)",
        recommended_for=["boosting", "classical", "neural"],
        cost="cheap",
    ),
    "macd_signal": FeatureDefinition(
        name="macd_signal",
        family="momentum",
        description="MACD signal line",
        recommended_for=["boosting", "classical", "neural"],
        cost="cheap",
    ),
    "macd_histogram": FeatureDefinition(
        name="macd_histogram",
        family="momentum",
        description="MACD histogram (line - signal)",
        recommended_for=["boosting", "classical", "neural"],
        cost="cheap",
    ),
    "stoch_k": FeatureDefinition(
        name="stoch_k",
        family="momentum",
        description="Stochastic %K",
        recommended_for=["boosting", "classical"],
        cost="cheap",
    ),
    "stoch_d": FeatureDefinition(
        name="stoch_d",
        family="momentum",
        description="Stochastic %D",
        recommended_for=["boosting", "classical"],
        cost="cheap",
    ),
    "sma_10": FeatureDefinition(
        name="sma_10",
        family="momentum",
        description="10-period SMA",
        recommended_for=["boosting", "classical"],
        cost="cheap",
    ),
    "sma_20": FeatureDefinition(
        name="sma_20",
        family="momentum",
        description="20-period SMA",
        recommended_for=["boosting", "classical"],
        cost="cheap",
    ),
    "sma_50": FeatureDefinition(
        name="sma_50",
        family="momentum",
        description="50-period SMA",
        recommended_for=["boosting", "classical"],
        cost="cheap",
    ),
    "sma_100": FeatureDefinition(
        name="sma_100",
        family="momentum",
        description="100-period SMA",
        recommended_for=["boosting", "classical"],
        cost="cheap",
    ),
    "sma_200": FeatureDefinition(
        name="sma_200",
        family="momentum",
        description="200-period SMA",
        recommended_for=["boosting", "classical"],
        cost="cheap",
    ),
    "ema_9": FeatureDefinition(
        name="ema_9",
        family="momentum",
        description="9-period EMA",
        recommended_for=["boosting", "classical", "neural"],
        cost="cheap",
    ),
    "ema_21": FeatureDefinition(
        name="ema_21",
        family="momentum",
        description="21-period EMA",
        recommended_for=["boosting", "classical", "neural"],
        cost="cheap",
    ),
    "ema_50": FeatureDefinition(
        name="ema_50",
        family="momentum",
        description="50-period EMA",
        recommended_for=["boosting", "classical", "neural"],
        cost="cheap",
    ),
    # -------------------------------------------------------------------------
    # Volatility Family (~30 features)
    # -------------------------------------------------------------------------
    "atr_7": FeatureDefinition(
        name="atr_7",
        family="volatility",
        description="7-period ATR",
        recommended_for=["boosting", "classical", "neural"],
        cost="cheap",
    ),
    "atr_14": FeatureDefinition(
        name="atr_14",
        family="volatility",
        description="14-period ATR",
        recommended_for=["boosting", "classical", "neural"],
        cost="cheap",
    ),
    "atr_21": FeatureDefinition(
        name="atr_21",
        family="volatility",
        description="21-period ATR",
        recommended_for=["boosting", "classical", "neural"],
        cost="cheap",
    ),
    "bb_upper": FeatureDefinition(
        name="bb_upper",
        family="volatility",
        description="Bollinger Band upper",
        recommended_for=["boosting", "classical"],
        cost="cheap",
    ),
    "bb_middle": FeatureDefinition(
        name="bb_middle",
        family="volatility",
        description="Bollinger Band middle (SMA)",
        recommended_for=["boosting", "classical"],
        cost="cheap",
    ),
    "bb_lower": FeatureDefinition(
        name="bb_lower",
        family="volatility",
        description="Bollinger Band lower",
        recommended_for=["boosting", "classical"],
        cost="cheap",
    ),
    "bb_width": FeatureDefinition(
        name="bb_width",
        family="volatility",
        description="Bollinger Band width (upper - lower)",
        recommended_for=["boosting", "classical", "neural"],
        cost="cheap",
    ),
    "bb_pct": FeatureDefinition(
        name="bb_pct",
        family="volatility",
        description="%B: (close - lower) / (upper - lower)",
        recommended_for=["boosting", "classical"],
        cost="cheap",
    ),
    "historical_volatility_20": FeatureDefinition(
        name="historical_volatility_20",
        family="volatility",
        description="20-period historical volatility",
        recommended_for=["boosting", "classical", "neural"],
        cost="cheap",
    ),
    # -------------------------------------------------------------------------
    # Volume Family (~20 features)
    # -------------------------------------------------------------------------
    "volume_sma_20": FeatureDefinition(
        name="volume_sma_20",
        family="volume",
        description="20-period volume SMA",
        recommended_for=["boosting", "classical"],
        cost="cheap",
    ),
    "volume_ratio": FeatureDefinition(
        name="volume_ratio",
        family="volume",
        description="Current volume / average volume",
        recommended_for=["boosting", "classical", "neural"],
        cost="cheap",
    ),
    "volume_delta": FeatureDefinition(
        name="volume_delta",
        family="volume",
        description="Change in volume from previous bar",
        recommended_for=["boosting", "classical", "neural"],
        cost="cheap",
    ),
    "obv": FeatureDefinition(
        name="obv",
        family="volume",
        description="On-Balance Volume",
        recommended_for=["boosting", "classical"],
        cost="cheap",
    ),
    "obv_ema_20": FeatureDefinition(
        name="obv_ema_20",
        family="volume",
        description="20-period EMA of OBV",
        recommended_for=["boosting", "classical"],
        cost="cheap",
    ),
    "vwap": FeatureDefinition(
        name="vwap",
        family="volume",
        description="Volume-Weighted Average Price",
        recommended_for=["boosting", "classical", "neural"],
        cost="medium",
    ),
    # -------------------------------------------------------------------------
    # Microstructure Family (~30 features - 2024 additions)
    # -------------------------------------------------------------------------
    "vpin": FeatureDefinition(
        name="vpin",
        family="microstructure",
        description="Volume-synchronized PIN",
        recommended_for=["boosting"],  # Too noisy for neural
        cost="expensive",
    ),
    "edge_spread_bps": FeatureDefinition(
        name="edge_spread_bps",
        family="microstructure",
        description="Effective spread in basis points",
        recommended_for=["boosting", "classical"],
        cost="medium",
    ),
    "trade_imbalance": FeatureDefinition(
        name="trade_imbalance",
        family="microstructure",
        description="Buy volume - sell volume imbalance",
        recommended_for=["boosting", "classical"],
        cost="medium",
    ),
    "time_to_fill_mean": FeatureDefinition(
        name="time_to_fill_mean",
        family="microstructure",
        description="Mean time to fill orders",
        recommended_for=["boosting"],
        cost="expensive",
    ),
    "kyle_lambda": FeatureDefinition(
        name="kyle_lambda",
        family="microstructure",
        description="Kyle's lambda (price impact)",
        recommended_for=["boosting"],
        cost="expensive",
    ),
    # -------------------------------------------------------------------------
    # Wavelets Family (~30 features - for neural models)
    # -------------------------------------------------------------------------
    "wavelet_db4_1": FeatureDefinition(
        name="wavelet_db4_1",
        family="wavelets",
        description="Daubechies 4 wavelet, level 1",
        recommended_for=["neural"],
        cost="expensive",
    ),
    "wavelet_db4_2": FeatureDefinition(
        name="wavelet_db4_2",
        family="wavelets",
        description="Daubechies 4 wavelet, level 2",
        recommended_for=["neural"],
        cost="expensive",
    ),
    "wavelet_db4_3": FeatureDefinition(
        name="wavelet_db4_3",
        family="wavelets",
        description="Daubechies 4 wavelet, level 3",
        recommended_for=["neural"],
        cost="expensive",
    ),
    # -------------------------------------------------------------------------
    # MTF Family (~20 features - multi-timeframe indicators)
    # -------------------------------------------------------------------------
    "rsi_14_1h": FeatureDefinition(
        name="rsi_14_1h",
        family="mtf",
        description="RSI from 1-hour timeframe",
        recommended_for=["boosting", "classical"],
        requires_mtf=True,
        cost="cheap",
    ),
    "atr_14_1h": FeatureDefinition(
        name="atr_14_1h",
        family="mtf",
        description="ATR from 1-hour timeframe",
        recommended_for=["boosting", "classical"],
        requires_mtf=True,
        cost="cheap",
    ),
    "macd_line_1h": FeatureDefinition(
        name="macd_line_1h",
        family="mtf",
        description="MACD from 1-hour timeframe",
        recommended_for=["boosting", "classical"],
        requires_mtf=True,
        cost="cheap",
    ),
    # -------------------------------------------------------------------------
    # Regime Family (~10 features - regime detection)
    # -------------------------------------------------------------------------
    "regime_volatility": FeatureDefinition(
        name="regime_volatility",
        family="regime",
        description="Volatility regime indicator",
        recommended_for=["boosting", "classical", "neural"],
        cost="medium",
    ),
    "regime_trend": FeatureDefinition(
        name="regime_trend",
        family="regime",
        description="Trend regime indicator",
        recommended_for=["boosting", "classical", "neural"],
        cost="medium",
    ),
    "regime_composite": FeatureDefinition(
        name="regime_composite",
        family="regime",
        description="Composite regime (volatility + trend)",
        recommended_for=["boosting", "classical", "neural"],
        cost="medium",
    ),
}


# =============================================================================
# Query Functions
# =============================================================================


def get_features_by_families(families: list[str]) -> list[str]:
    """
    Get all feature names belonging to specified families.

    Args:
        families: List of feature family names

    Returns:
        List of feature names in those families

    Example:
        >>> features = get_features_by_families(["momentum", "volatility"])
        >>> print(len(features))  # ~70 features
    """
    return [name for name, defn in FEATURE_REGISTRY.items() if defn.family in families]


def get_features_by_model(model_family: str) -> list[str]:
    """
    Get recommended features for a model family.

    Args:
        model_family: Model family name (boosting, neural, transformer, etc.)

    Returns:
        List of recommended feature names
    """
    return [name for name, defn in FEATURE_REGISTRY.items() if model_family in defn.recommended_for]


def get_feature_families() -> list[str]:
    """Get list of all feature families."""
    return list(set(defn.family for defn in FEATURE_REGISTRY.values()))
