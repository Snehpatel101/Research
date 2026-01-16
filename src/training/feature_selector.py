"""Feature selection based on configuration mode."""

import logging
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


class FeatureSelector:
    """Selects features based on mode and MTF strategy."""

    def __init__(self, feature_config_path: str | Path = "config/features/day_trading.yaml"):
        """Initialize with feature config."""
        self.config_path = Path(feature_config_path)
        self.feature_config = self._load_feature_config()

    def _load_feature_config(self) -> dict:
        """Load feature configuration from YAML."""
        if not self.config_path.exists():
            logger.warning(f"Feature config not found: {self.config_path}, using defaults")
            return self._default_config()

        with open(self.config_path) as f:
            return yaml.safe_load(f)

    def _default_config(self) -> dict:
        """Default feature configuration if file missing."""
        return {
            "modes": {
                "full": {"groups": ["all"]},
                "hft_only": {"groups": ["microstructure_2024"]},
                "classical": {"groups": ["momentum", "volatility", "volume", "trend"]},
                "minimal": {"groups": ["momentum", "volatility"]},
                "research_2024": {"groups": ["microstructure_2024", "momentum", "volatility"]},
            }
        }

    def select_features(
        self,
        df: pd.DataFrame,
        mode: str = "full",
        mtf_strategy: str = "indicators",
    ) -> pd.DataFrame:
        """
        Select features based on mode and MTF strategy.

        Args:
            df: DataFrame with all features
            mode: Feature mode (full/hft_only/classical/minimal/research_2024)
            mtf_strategy: MTF strategy (none/indicators/bars/multi_stream)

        Returns:
            DataFrame with selected features only
        """
        logger.info(f"Selecting features with mode={mode}, mtf_strategy={mtf_strategy}")

        if mode not in self.feature_config.get("modes", {}):
            logger.warning(f"Unknown feature mode: {mode}, using 'full'")
            mode = "full"

        mode_config = self.feature_config["modes"][mode]
        feature_groups = mode_config.get("groups", [])

        if "all" in feature_groups or feature_groups == "all":
            logger.info(f"Using all {len(df.columns)} features")
            return df

        selected_cols = self._select_columns_by_groups(df, feature_groups)

        if mtf_strategy == "none":
            selected_cols = [c for c in selected_cols if "mtf_" not in c.lower()]
        elif mtf_strategy == "indicators":
            pass
        elif mtf_strategy == "bars":
            mtf_bar_cols = [
                c
                for c in df.columns
                if "mtf_" in c and any(x in c for x in ["open", "high", "low", "close", "volume"])
            ]
            selected_cols.extend(mtf_bar_cols)

        selected_cols = list(set(selected_cols))
        selected_cols = [c for c in selected_cols if c in df.columns]

        logger.info(f"Selected {len(selected_cols)} features from mode '{mode}'")

        return df[selected_cols]

    def _select_columns_by_groups(self, df: pd.DataFrame, groups: list[str]) -> list[str]:
        """Select columns matching feature groups."""
        group_patterns = {
            "microstructure_2024": [
                "edge_spread",
                "roll_spread",
                "time_to",
                "vpin",
                "kyle_lambda",
                "amihud_illiquidity",
                "body_size",
                "wick",
                "bar_range",
                "bar_direction",
                "volume_density",
                "volume_concentration",
                "intrabar_reversal",
            ],
            "microstructure_classical": [
                "micro_amihud",
                "micro_roll",
                "micro_kyle",
                "micro_cs",
                "micro_rel",
                "micro_volume_imbalance",
                "micro_trade_intensity",
                "micro_efficiency",
                "micro_vol_ratio",
            ],
            "momentum": ["rsi", "macd", "mfi", "roc", "cci", "stochastic", "williams"],
            "volatility": [
                "atr",
                "bb_",
                "keltner",
                "volatility",
                "parkinson",
                "garman_klass",
                "rogers_satchell",
                "yang_zhang",
            ],
            "volume": ["volume_sma", "volume_ratio", "obv", "vwap", "dollar_volume", "twap"],
            "trend": ["adx", "di_plus", "di_minus", "supertrend"],
            "wavelets": ["wavelet_"],
            "mtf_indicators": ["mtf_"],
            "regime": ["regime_"],
            "temporal": ["hour_", "minute_", "day_of_week"],
        }

        selected = []

        for group in groups:
            if group in group_patterns:
                patterns = group_patterns[group]
                for col in df.columns:
                    if any(pattern in col.lower() for pattern in patterns):
                        selected.append(col)

        return selected

    def get_feature_count(self, mode: str) -> int:
        """Get expected feature count for mode."""
        counts = {
            "full": 210,
            "hft_only": 20,
            "classical": 150,
            "minimal": 60,
            "research_2024": 100,
        }
        return counts.get(mode, 210)


__all__ = ["FeatureSelector"]
