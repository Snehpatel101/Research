from dataclasses import dataclass, field


@dataclass
class ModelFeatureStrategy:
    model_name: str
    family: str
    baseline_features: list[str]
    preferred_families: list[str] = field(default_factory=list)
    mtf_mode: str = "none"
    requires_sequences: bool = False
    min_features: int = 20
    max_features: int = 200

    def __repr__(self) -> str:
        return (
            f"ModelFeatureStrategy({self.model_name}, "
            f"family={self.family}, "
            f"baseline={len(self.baseline_features)} features, "
            f"mtf={self.mtf_mode})"
        )


BASELINE_MOMENTUM = [
    "returns",
    "returns_squared",
    "log_returns",
    "rsi_14",
    "macd_line",
    "macd_signal",
    "macd_histogram",
    "stoch_k",
    "stoch_d",
    "williams_r_14",
    "roc_10",
    "roc_20",
    "cci_20",
    "mfi_14",
    "sma_10",
    "sma_20",
    "sma_50",
    "sma_100",
    "sma_200",
    "ema_9",
    "ema_21",
    "ema_50",
]

BASELINE_VOLATILITY = [
    "atr_7",
    "atr_14",
    "atr_21",
    "bb_upper",
    "bb_middle",
    "bb_lower",
    "bb_width",
    "bb_pct",
    "kc_upper",
    "kc_middle",
    "kc_lower",
    "historical_volatility_20",
    "historical_volatility_50",
    "parkinson_vol_20",
    "garman_klass_vol_20",
    "rogers_satchell_vol_20",
    "yang_zhang_vol_20",
]

BASELINE_VOLUME = [
    "volume",
    "volume_sma_20",
    "volume_ratio",
    "volume_delta",
    "obv",
    "obv_ema_20",
    "vwap",
    "dollar_volume",
]

BASELINE_TREND = [
    "adx_14",
    "adx_pos_di",
    "adx_neg_di",
    "supertrend",
    "supertrend_direction",
]

BASELINE_PRICE = [
    "open",
    "high",
    "low",
    "close",
    "high_low_ratio",
    "close_open_ratio",
    "clv",
]

BASELINE_MICROSTRUCTURE = [
    "vpin",
    "edge_spread_bps",
    "trade_imbalance",
    "kyle_lambda",
    "amihud_illiquidity",
]

BASELINE_WAVELETS = [
    "wavelet_db4_d1",
    "wavelet_db4_d2",
    "wavelet_db4_d3",
    "wavelet_db4_a3",
]

BASELINE_REGIME = [
    "regime_volatility",
    "regime_trend",
    "regime_composite",
]

BASELINE_TEMPORAL = [
    "hour_sin",
    "hour_cos",
    "day_of_week_sin",
    "day_of_week_cos",
]

MTF_INDICATORS_15MIN = [
    "rsi_14_15min",
    "atr_14_15min",
    "macd_line_15min",
    "bb_width_15min",
    "adx_14_15min",
]

MTF_INDICATORS_1H = [
    "rsi_14_1h",
    "atr_14_1h",
    "macd_line_1h",
    "bb_width_1h",
    "adx_14_1h",
    "sma_50_1h",
    "ema_21_1h",
]


MODEL_FEATURE_STRATEGIES = {
    "xgboost": ModelFeatureStrategy(
        model_name="xgboost",
        family="boosting",
        baseline_features=(
            BASELINE_MOMENTUM
            + BASELINE_VOLATILITY
            + BASELINE_VOLUME
            + BASELINE_TREND
            + BASELINE_PRICE
            + BASELINE_MICROSTRUCTURE
            + MTF_INDICATORS_15MIN
            + MTF_INDICATORS_1H
        ),
        preferred_families=["momentum", "volatility", "volume", "microstructure", "mtf"],
        mtf_mode="indicators",
        min_features=40,
        max_features=120,
    ),
    "lightgbm": ModelFeatureStrategy(
        model_name="lightgbm",
        family="boosting",
        baseline_features=(
            BASELINE_MOMENTUM
            + BASELINE_VOLATILITY
            + BASELINE_VOLUME
            + BASELINE_TREND
            + BASELINE_MICROSTRUCTURE
            + MTF_INDICATORS_15MIN
            + MTF_INDICATORS_1H
        ),
        preferred_families=["momentum", "volatility", "volume", "microstructure", "mtf"],
        mtf_mode="indicators",
        min_features=40,
        max_features=120,
    ),
    "catboost": ModelFeatureStrategy(
        model_name="catboost",
        family="boosting",
        baseline_features=(
            BASELINE_MOMENTUM
            + BASELINE_VOLATILITY
            + BASELINE_VOLUME
            + BASELINE_TREND
            + BASELINE_MICROSTRUCTURE
            + MTF_INDICATORS_15MIN
            + MTF_INDICATORS_1H
        ),
        preferred_families=["momentum", "volatility", "volume", "microstructure", "mtf"],
        mtf_mode="indicators",
        min_features=40,
        max_features=120,
    ),
    "random_forest": ModelFeatureStrategy(
        model_name="random_forest",
        family="classical",
        baseline_features=(
            BASELINE_MOMENTUM
            + BASELINE_VOLATILITY
            + BASELINE_VOLUME
            + BASELINE_TREND
            + MTF_INDICATORS_15MIN
        ),
        preferred_families=["momentum", "volatility", "volume"],
        mtf_mode="indicators",
        min_features=30,
        max_features=100,
    ),
    "logistic": ModelFeatureStrategy(
        model_name="logistic",
        family="classical",
        baseline_features=(BASELINE_MOMENTUM[:10] + BASELINE_VOLATILITY[:8] + BASELINE_VOLUME[:5]),
        preferred_families=["momentum", "volatility"],
        mtf_mode="none",
        min_features=15,
        max_features=50,
    ),
    "svm": ModelFeatureStrategy(
        model_name="svm",
        family="classical",
        baseline_features=(BASELINE_MOMENTUM[:10] + BASELINE_VOLATILITY[:8] + BASELINE_VOLUME[:5]),
        preferred_families=["momentum", "volatility"],
        mtf_mode="none",
        min_features=15,
        max_features=50,
    ),
    "lstm": ModelFeatureStrategy(
        model_name="lstm",
        family="neural",
        baseline_features=(
            BASELINE_MOMENTUM
            + BASELINE_VOLATILITY
            + BASELINE_VOLUME
            + BASELINE_WAVELETS
            + MTF_INDICATORS_15MIN
        ),
        preferred_families=["momentum", "volatility", "wavelets", "mtf"],
        mtf_mode="indicators",
        requires_sequences=True,
        min_features=50,
        max_features=150,
    ),
    "gru": ModelFeatureStrategy(
        model_name="gru",
        family="neural",
        baseline_features=(
            BASELINE_MOMENTUM
            + BASELINE_VOLATILITY
            + BASELINE_VOLUME
            + BASELINE_WAVELETS
            + MTF_INDICATORS_15MIN
        ),
        preferred_families=["momentum", "volatility", "wavelets", "mtf"],
        mtf_mode="indicators",
        requires_sequences=True,
        min_features=50,
        max_features=150,
    ),
    "tcn": ModelFeatureStrategy(
        model_name="tcn",
        family="neural",
        baseline_features=(
            BASELINE_MOMENTUM + BASELINE_VOLATILITY + BASELINE_VOLUME + BASELINE_WAVELETS
        ),
        preferred_families=["momentum", "volatility", "volume", "wavelets"],
        mtf_mode="none",
        requires_sequences=True,
        min_features=50,
        max_features=120,
    ),
    "transformer": ModelFeatureStrategy(
        model_name="transformer",
        family="transformer",
        baseline_features=BASELINE_PRICE,
        preferred_families=["raw"],
        mtf_mode="multi_stream",
        requires_sequences=True,
        min_features=4,
        max_features=20,
    ),
    "patchtst": ModelFeatureStrategy(
        model_name="patchtst",
        family="transformer",
        baseline_features=BASELINE_PRICE + ["volume"],
        preferred_families=["raw"],
        mtf_mode="multi_stream",
        requires_sequences=True,
        min_features=4,
        max_features=10,
    ),
    "itransformer": ModelFeatureStrategy(
        model_name="itransformer",
        family="transformer",
        baseline_features=BASELINE_PRICE + ["volume"],
        preferred_families=["raw"],
        mtf_mode="multi_stream",
        requires_sequences=True,
        min_features=4,
        max_features=10,
    ),
    "tft": ModelFeatureStrategy(
        model_name="tft",
        family="transformer",
        baseline_features=(BASELINE_PRICE + BASELINE_MOMENTUM[:10] + BASELINE_VOLATILITY[:5]),
        preferred_families=["raw", "momentum", "volatility"],
        mtf_mode="indicators",
        requires_sequences=True,
        min_features=10,
        max_features=40,
    ),
    "nbeats": ModelFeatureStrategy(
        model_name="nbeats",
        family="transformer",
        baseline_features=["close", "volume"],
        preferred_families=["raw"],
        mtf_mode="none",
        requires_sequences=True,
        min_features=2,
        max_features=10,
    ),
    "inceptiontime": ModelFeatureStrategy(
        model_name="inceptiontime",
        family="cnn",
        baseline_features=(BASELINE_MOMENTUM + BASELINE_VOLATILITY + BASELINE_VOLUME),
        preferred_families=["momentum", "volatility", "volume"],
        mtf_mode="none",
        requires_sequences=True,
        min_features=30,
        max_features=100,
    ),
    "resnet1d": ModelFeatureStrategy(
        model_name="resnet1d",
        family="cnn",
        baseline_features=(BASELINE_MOMENTUM + BASELINE_VOLATILITY + BASELINE_VOLUME),
        preferred_families=["momentum", "volatility", "volume"],
        mtf_mode="none",
        requires_sequences=True,
        min_features=30,
        max_features=100,
    ),
}

for meta_model in ["ridge_meta", "mlp_meta", "calibrated_meta", "xgboost_meta"]:
    MODEL_FEATURE_STRATEGIES[meta_model] = ModelFeatureStrategy(
        model_name=meta_model,
        family="meta_learner",
        baseline_features=[],
        preferred_families=[],
        mtf_mode="none",
        min_features=2,
        max_features=20,
    )

for ensemble in ["voting", "stacking", "blending"]:
    MODEL_FEATURE_STRATEGIES[ensemble] = ModelFeatureStrategy(
        model_name=ensemble,
        family="ensemble",
        baseline_features=[],
        preferred_families=[],
        mtf_mode="none",
        min_features=10,
        max_features=200,
    )


def get_strategy_for_model(model_name: str) -> ModelFeatureStrategy:
    if model_name not in MODEL_FEATURE_STRATEGIES:
        raise ValueError(
            f"No strategy defined for model '{model_name}'. "
            f"Available: {list(MODEL_FEATURE_STRATEGIES.keys())}"
        )
    return MODEL_FEATURE_STRATEGIES[model_name]


def get_baseline_features(model_name: str) -> list[str]:
    strategy = get_strategy_for_model(model_name)
    return strategy.baseline_features.copy()
