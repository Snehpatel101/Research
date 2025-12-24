import pandas as pd

from src.phase1.config.feature_sets import FEATURE_SET_DEFINITIONS
from src.phase1.utils.feature_sets import resolve_feature_set, build_feature_set_manifest


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "datetime": pd.date_range("2024-01-01", periods=3, freq="min"),
            "symbol": ["MES"] * 3,
            "open": [1.0, 1.1, 1.2],
            "high": [1.1, 1.2, 1.3],
            "low": [0.9, 1.0, 1.1],
            "close": [1.05, 1.15, 1.25],
            "volume": [100, 110, 120],
            "rsi_14": [50, 55, 60],
            "macd_hist": [0.1, 0.2, 0.1],
            "sma_10": [1.0, 1.1, 1.2],
            "price_to_vwap": [1.0, 1.01, 0.99],
            "bb_position": [0.5, 0.4, 0.6],
            "trend_regime": [0, 1, 1],
            "volatility_regime": [1, 1, 0],
            "rsi_14_15m": [52, 56, 59],
            "close_1h": [1.0, 1.1, 1.2],
            "mes_mgc_beta": [0.0, 0.0, 0.0],
            "label_h5": [1, 0, -1],
            "sample_weight_h5": [1.0, 1.0, 1.0],
        }
    )


def test_feature_set_resolution():
    df = _sample_df()

    core_full = resolve_feature_set(df, FEATURE_SET_DEFINITIONS["core_full"])
    assert "rsi_14" in core_full
    assert "rsi_14_15m" not in core_full
    assert "mes_mgc_beta" not in core_full

    mtf_plus = resolve_feature_set(df, FEATURE_SET_DEFINITIONS["mtf_plus"])
    assert "rsi_14_15m" in mtf_plus
    assert "mes_mgc_beta" in mtf_plus

    core_min = resolve_feature_set(df, FEATURE_SET_DEFINITIONS["core_min"])
    assert "sma_10" not in core_min
    assert "rsi_14" in core_min


def test_feature_set_manifest_counts():
    df = _sample_df()
    manifest = build_feature_set_manifest(df, FEATURE_SET_DEFINITIONS)
    assert "core_min" in manifest
    assert manifest["mtf_plus"]["feature_count"] >= manifest["core_full"]["feature_count"]
