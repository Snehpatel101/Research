# PHASE 1: UNIFIED FEATURES - Single Feature Registry

**Status:** PLANNING
**Created:** 2026-01-16
**Purpose:** Create single feature registry with all 150+ indicators and per-model feature strategies

---

## Overview

Phase 1 creates a unified feature system where:
1. ALL 150+ features are registered in one place
2. Each of 22 models has a tailored baseline feature strategy
3. Optuna-based optimization prunes from baseline to optimal

---

## Task 1.1: Create FeatureRegistry

### File: `src/features/registry.py`

```python
from typing import Dict, List, Callable, Optional, Any
from dataclasses import dataclass, field
import pandas as pd

@dataclass
class FeatureDefinition:
    """Definition of a single feature."""
    name: str
    family: str
    compute_fn: Callable[[pd.DataFrame], pd.Series]
    dependencies: List[str] = field(default_factory=list)
    params: Dict[str, Any] = field(default_factory=dict)
    recommended_for: List[str] = field(default_factory=list)
    cost: str = "cheap"  # cheap, medium, expensive


class FeatureRegistry:
    """Singleton registry for all 150+ features."""

    _instance = None
    _features: Dict[str, FeatureDefinition] = {}
    _families: Dict[str, List[str]] = {}

    @classmethod
    def register(cls, name: str, family: str, dependencies: List[str] = None,
                 recommended_for: List[str] = None, cost: str = "cheap"):
        """Decorator to register feature computation."""
        def decorator(fn):
            cls._features[name] = FeatureDefinition(
                name=name, family=family, compute_fn=fn,
                dependencies=dependencies or [],
                recommended_for=recommended_for or [],
                cost=cost,
            )
            if family not in cls._families:
                cls._families[family] = []
            cls._families[family].append(name)
            return fn
        return decorator

    @classmethod
    def get_by_family(cls, family: str) -> List[str]:
        return cls._families.get(family, [])

    @classmethod
    def get_by_families(cls, families: List[str]) -> List[str]:
        result = []
        for family in families:
            result.extend(cls.get_by_family(family))
        return result

    @classmethod
    def compute(cls, df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
        result = df.copy()
        for name in features:
            if name in cls._features:
                result[name] = cls._features[name].compute_fn(df)
        return result

    @classmethod
    def list_all(cls) -> List[str]:
        return list(cls._features.keys())

    @classmethod
    def count(cls) -> int:
        return len(cls._features)
```

---

## Task 1.2: Define All 12 Feature Families (162 Total Features)

### FAMILY 1: RAW (5 features)
| Feature | Description |
|---------|-------------|
| `open` | Open price |
| `high` | High price |
| `low` | Low price |
| `close` | Close price |
| `volume` | Volume |

### FAMILY 2: MOMENTUM (23 features)
| Feature | Description |
|---------|-------------|
| `rsi_7` | 7-period RSI |
| `rsi_14` | 14-period RSI |
| `rsi_21` | 21-period RSI |
| `rsi_overbought` | RSI > 70 flag |
| `rsi_oversold` | RSI < 30 flag |
| `macd_line` | MACD line (12,26) |
| `macd_signal` | MACD signal |
| `macd_histogram` | MACD histogram |
| `macd_cross_up` | Bullish crossover |
| `macd_cross_down` | Bearish crossover |
| `stoch_k` | Stochastic %K |
| `stoch_d` | Stochastic %D |
| `stoch_overbought` | Stoch > 80 |
| `stoch_oversold` | Stoch < 20 |
| `williams_r` | Williams %R |
| `roc_5` | ROC 5-period |
| `roc_10` | ROC 10-period |
| `roc_20` | ROC 20-period |
| `cci_14` | CCI 14-period |
| `cci_20` | CCI 20-period |
| `mfi_14` | MFI 14-period |
| `mfi_overbought` | MFI > 80 |
| `mfi_oversold` | MFI < 20 |

### FAMILY 3: MOVING_AVERAGE (16 features)
| Feature | Description |
|---------|-------------|
| `sma_10` | SMA 10-period |
| `sma_20` | SMA 20-period |
| `sma_50` | SMA 50-period |
| `sma_100` | SMA 100-period |
| `sma_200` | SMA 200-period |
| `ema_9` | EMA 9-period |
| `ema_12` | EMA 12-period |
| `ema_21` | EMA 21-period |
| `ema_26` | EMA 26-period |
| `ema_50` | EMA 50-period |
| `price_to_sma_20` | Price/SMA-20 ratio |
| `price_to_sma_50` | Price/SMA-50 ratio |
| `price_to_ema_21` | Price/EMA-21 ratio |
| `sma_cross_10_50` | SMA 10/50 crossover |
| `sma_cross_20_200` | SMA 20/200 crossover |
| `ema_cross_9_21` | EMA 9/21 crossover |

### FAMILY 4: VOLATILITY (25 features)
| Feature | Description |
|---------|-------------|
| `atr_7` | ATR 7-period |
| `atr_14` | ATR 14-period |
| `atr_21` | ATR 21-period |
| `atr_pct_14` | ATR as % of price |
| `bb_upper` | Bollinger upper |
| `bb_middle` | Bollinger middle |
| `bb_lower` | Bollinger lower |
| `bb_width` | Bollinger width |
| `bb_position` | Price position in BB |
| `close_bb_zscore` | Close z-score vs BB |
| `kc_upper` | Keltner upper |
| `kc_middle` | Keltner middle |
| `kc_lower` | Keltner lower |
| `kc_position` | Price position in KC |
| `hvol_10` | Historical vol 10 |
| `hvol_20` | Historical vol 20 |
| `hvol_60` | Historical vol 60 |
| `parkinson_vol` | Parkinson volatility |
| `gk_vol` | Garman-Klass vol |
| `rs_vol` | Rogers-Satchell vol |
| `yz_vol` | Yang-Zhang vol |
| `return_skew_20` | Return skewness |
| `return_kurt_20` | Return kurtosis |
| `garch_vol_forecast` | GARCH forecast |
| `garch_vol_ratio` | GARCH/realized ratio |

### FAMILY 5: VOLUME (15 features)
| Feature | Description |
|---------|-------------|
| `obv` | On-Balance Volume |
| `obv_sma_20` | OBV 20-period SMA |
| `volume_sma_20` | Volume 20-period SMA |
| `volume_ratio` | Volume / 20-period avg |
| `volume_zscore` | Volume z-score |
| `vwap` | Session VWAP |
| `price_to_vwap` | Price/VWAP deviation |
| `twap` | TWAP |
| `twap_10` | Rolling TWAP 10 |
| `twap_20` | Rolling TWAP 20 |
| `price_to_twap_10` | Close/TWAP ratio |
| `dollar_volume` | Price x Volume |
| `dollar_volume_sma_10` | Dollar vol SMA 10 |
| `dollar_volume_sma_20` | Dollar vol SMA 20 |
| `dollar_volume_ratio` | Dollar vol ratio |

### FAMILY 6: TREND (6 features)
| Feature | Description |
|---------|-------------|
| `adx_14` | ADX 14-period |
| `plus_di_14` | +DI 14-period |
| `minus_di_14` | -DI 14-period |
| `adx_strong_trend` | ADX > 25 flag |
| `supertrend` | Supertrend (10, 3.0) |
| `supertrend_direction` | Supertrend direction |

### FAMILY 7: PRICE (12 features)
| Feature | Description |
|---------|-------------|
| `return_1` | 1-period return |
| `return_5` | 5-period return |
| `return_10` | 10-period return |
| `return_20` | 20-period return |
| `log_return_1` | Log return 1 |
| `log_return_5` | Log return 5 |
| `hl_ratio` | High/Low ratio |
| `co_ratio` | Close/Open ratio |
| `range_pct` | Range % of close |
| `clv` | Close Location Value |
| `autocorr_lag1` | Autocorr lag-1 |
| `autocorr_lag5` | Autocorr lag-5 |

### FAMILY 8: MICROSTRUCTURE (15 features)
| Feature | Description |
|---------|-------------|
| `micro_amihud` | Amihud illiquidity |
| `micro_amihud_10` | Amihud 10-avg |
| `micro_amihud_20` | Amihud 20-avg |
| `micro_roll_spread` | Roll spread |
| `micro_roll_spread_pct` | Roll spread % |
| `micro_kyle_lambda` | Kyle's lambda |
| `micro_cs_spread` | Corwin-Schultz spread |
| `micro_rel_spread` | Relative spread |
| `micro_rel_spread_10` | Rel spread 10-avg |
| `micro_volume_imbalance` | Volume imbalance |
| `micro_cum_imbalance_20` | Cumulative imbalance |
| `micro_trade_intensity_20` | Trade intensity |
| `micro_trade_intensity_50` | Trade intensity 50 |
| `micro_efficiency_10` | Price efficiency |
| `micro_vol_ratio` | Realized vol ratio |

### FAMILY 9: ENTROPY (12 features)
| Feature | Description |
|---------|-------------|
| `entropy_shannon_10` | Shannon entropy 10 |
| `entropy_shannon_20` | Shannon entropy 20 |
| `entropy_shannon_50` | Shannon entropy 50 |
| `entropy_shannon_norm_20` | Normalized Shannon |
| `entropy_lz_20` | Lempel-Ziv 20 |
| `entropy_lz_50` | Lempel-Ziv 50 |
| `entropy_apen_20` | Approx entropy 20 |
| `entropy_apen_50` | Approx entropy 50 |
| `sample_entropy_20` | Sample entropy 20 |
| `hurst_50` | Hurst exponent 50 |
| `hurst_100` | Hurst exponent 100 |
| `hurst_regime` | Hurst regime |

### FAMILY 10: WAVELETS (15 features)
| Feature | Description |
|---------|-------------|
| `wavelet_close_approx` | DWT approximation |
| `wavelet_close_d1` | DWT detail 1 |
| `wavelet_close_d2` | DWT detail 2 |
| `wavelet_close_d3` | DWT detail 3 |
| `wavelet_volume_approx` | Volume DWT approx |
| `wavelet_volume_d1` | Volume DWT d1 |
| `wavelet_close_energy_approx` | Wavelet energy approx |
| `wavelet_close_energy_d1` | Wavelet energy d1 |
| `wavelet_close_energy_d2` | Wavelet energy d2 |
| `wavelet_close_energy_d3` | Wavelet energy d3 |
| `wavelet_close_energy_ratio` | Energy ratio |
| `wavelet_close_volatility` | Wavelet volatility |
| `wavelet_close_trend_strength` | Wavelet trend strength |
| `wavelet_close_trend_direction` | Wavelet trend dir |
| `wavelet_volume_energy_ratio` | Vol energy ratio |

### FAMILY 11: TEMPORAL (9 features)
| Feature | Description |
|---------|-------------|
| `hour_sin` | Hour sine encoding |
| `hour_cos` | Hour cosine encoding |
| `minute_sin` | Minute sine encoding |
| `minute_cos` | Minute cosine encoding |
| `dayofweek_sin` | Day-of-week sine |
| `dayofweek_cos` | Day-of-week cosine |
| `session_asia` | Asia session flag |
| `session_london` | London session flag |
| `session_ny` | NY session flag |

### FAMILY 12: REGIME (9 features)
| Feature | Description |
|---------|-------------|
| `volatility_regime` | Vol regime (0/1) |
| `trend_regime` | Trend regime (-1/0/1) |
| `structure_regime` | Market structure |
| `regime_low_vol` | Low vol flag |
| `regime_high_vol` | High vol flag |
| `regime_uptrend` | Uptrend flag |
| `regime_downtrend` | Downtrend flag |
| `regime_sideways` | Sideways flag |
| `composite_regime` | 9-combination |

### FAMILY 13: MTF (30+ per TF)
For each of 9 timeframes, replicate key indicators with `_{tf}` suffix:
- `rsi_14_5min`, `rsi_14_15min`, `rsi_14_1h`
- `atr_14_5min`, `atr_14_15min`, `atr_14_1h`
- `macd_line_5min`, `macd_line_15min`, `macd_line_1h`
- All use `shift(1)` for anti-lookahead

---

## Task 1.3: Model Feature Strategies (22 Models)

### File: `src/features/strategies.py`

```python
from dataclasses import dataclass
from typing import List, Literal

MTFMode = Literal["none", "indicators", "multi_stream"]

@dataclass
class ModelFeatureStrategy:
    model_name: str
    family: str
    baseline_families: List[str]
    min_features: int = 20
    max_features: int = 200
    mtf_mode: MTFMode = "none"
    requires_sequences: bool = False


MODEL_FEATURE_STRATEGIES = {
    # BOOSTING (3) - Full engineered + MTF
    "xgboost": ModelFeatureStrategy(
        model_name="xgboost", family="boosting",
        baseline_families=["momentum", "volatility", "volume", "trend", "microstructure", "mtf"],
        min_features=40, max_features=150, mtf_mode="indicators",
    ),
    "lightgbm": ModelFeatureStrategy(
        model_name="lightgbm", family="boosting",
        baseline_families=["momentum", "volatility", "volume", "trend", "microstructure", "mtf"],
        min_features=40, max_features=150, mtf_mode="indicators",
    ),
    "catboost": ModelFeatureStrategy(
        model_name="catboost", family="boosting",
        baseline_families=["momentum", "volatility", "volume", "trend", "microstructure", "mtf"],
        min_features=40, max_features=150, mtf_mode="indicators",
    ),

    # CLASSICAL (3) - Simpler features
    "random_forest": ModelFeatureStrategy(
        model_name="random_forest", family="classical",
        baseline_families=["momentum", "volatility", "volume", "trend"],
        min_features=30, max_features=100, mtf_mode="indicators",
    ),
    "logistic": ModelFeatureStrategy(
        model_name="logistic", family="classical",
        baseline_families=["momentum", "volatility"],
        min_features=15, max_features=50, mtf_mode="none",
    ),
    "svm": ModelFeatureStrategy(
        model_name="svm", family="classical",
        baseline_families=["momentum", "volatility"],
        min_features=15, max_features=50, mtf_mode="none",
    ),

    # NEURAL RNN (3) - Temporal patterns
    "lstm": ModelFeatureStrategy(
        model_name="lstm", family="neural",
        baseline_families=["momentum", "volatility", "volume", "wavelets"],
        min_features=50, max_features=150, mtf_mode="indicators", requires_sequences=True,
    ),
    "gru": ModelFeatureStrategy(
        model_name="gru", family="neural",
        baseline_families=["momentum", "volatility", "volume", "wavelets"],
        min_features=50, max_features=150, mtf_mode="indicators", requires_sequences=True,
    ),
    "tcn": ModelFeatureStrategy(
        model_name="tcn", family="neural",
        baseline_families=["momentum", "volatility", "volume", "wavelets"],
        min_features=50, max_features=120, mtf_mode="none", requires_sequences=True,
    ),

    # TRANSFORMER (4) - Raw OHLCV multi-stream
    "transformer": ModelFeatureStrategy(
        model_name="transformer", family="transformer",
        baseline_families=["raw"],
        min_features=4, max_features=20, mtf_mode="multi_stream", requires_sequences=True,
    ),
    "patchtst": ModelFeatureStrategy(
        model_name="patchtst", family="transformer",
        baseline_families=["raw"],
        min_features=4, max_features=10, mtf_mode="multi_stream", requires_sequences=True,
    ),
    "itransformer": ModelFeatureStrategy(
        model_name="itransformer", family="transformer",
        baseline_families=["raw"],
        min_features=4, max_features=10, mtf_mode="multi_stream", requires_sequences=True,
    ),
    "tft": ModelFeatureStrategy(
        model_name="tft", family="transformer",
        baseline_families=["raw", "momentum", "volatility"],
        min_features=10, max_features=40, mtf_mode="indicators", requires_sequences=True,
    ),

    # OTHER NEURAL (3)
    "nbeats": ModelFeatureStrategy(
        model_name="nbeats", family="neural",
        baseline_families=["raw"],
        min_features=2, max_features=10, mtf_mode="none", requires_sequences=True,
    ),
    "inceptiontime": ModelFeatureStrategy(
        model_name="inceptiontime", family="cnn",
        baseline_families=["momentum", "volatility", "volume"],
        min_features=30, max_features=100, mtf_mode="none", requires_sequences=True,
    ),
    "resnet1d": ModelFeatureStrategy(
        model_name="resnet1d", family="cnn",
        baseline_families=["momentum", "volatility", "volume"],
        min_features=30, max_features=100, mtf_mode="none", requires_sequences=True,
    ),

    # META-LEARNERS (4) - OOF predictions only
    "ridge_meta": ModelFeatureStrategy(
        model_name="ridge_meta", family="meta_learner",
        baseline_families=[], min_features=2, max_features=20,
    ),
    "mlp_meta": ModelFeatureStrategy(
        model_name="mlp_meta", family="meta_learner",
        baseline_families=[], min_features=2, max_features=20,
    ),
    "calibrated_meta": ModelFeatureStrategy(
        model_name="calibrated_meta", family="meta_learner",
        baseline_families=[], min_features=2, max_features=20,
    ),
    "xgboost_meta": ModelFeatureStrategy(
        model_name="xgboost_meta", family="meta_learner",
        baseline_families=[], min_features=2, max_features=20,
    ),

    # ENSEMBLE (3) - Inherit from bases
    "voting": ModelFeatureStrategy(
        model_name="voting", family="ensemble",
        baseline_families=[], min_features=10, max_features=200,
    ),
    "stacking": ModelFeatureStrategy(
        model_name="stacking", family="ensemble",
        baseline_families=[], min_features=10, max_features=200,
    ),
    "blending": ModelFeatureStrategy(
        model_name="blending", family="ensemble",
        baseline_families=[], min_features=10, max_features=200,
    ),
}


def get_strategy_for_model(model_name: str) -> ModelFeatureStrategy:
    if model_name not in MODEL_FEATURE_STRATEGIES:
        raise ValueError(f"No strategy for model '{model_name}'")
    return MODEL_FEATURE_STRATEGIES[model_name]
```

---

## Task 1.4: Feature Optimization with Optuna

### File: `src/features/optimization.py`

```python
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import optuna

from .strategies import get_strategy_for_model
from .registry import FeatureRegistry


@dataclass
class OptimizationResult:
    model_name: str
    original_features: List[str]
    optimized_features: List[str]
    n_original: int
    n_optimized: int
    improvement: float
    best_score: float
    n_trials: int


class FeatureOptimizer:
    """Optuna-based feature pruning from baseline to optimal."""

    def __init__(self, model_name: str, n_trials: int = 50, metric: str = "f1_weighted"):
        self.model_name = model_name
        self.n_trials = n_trials
        self.metric = metric
        self._strategy = get_strategy_for_model(model_name)

    def optimize(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: List[str],
    ) -> OptimizationResult:
        """Prune baseline features to optimal subset."""
        from src.models.registry import ModelRegistry
        from sklearn.metrics import f1_score

        min_features = self._strategy.min_features

        if len(feature_names) <= min_features:
            return OptimizationResult(
                model_name=self.model_name,
                original_features=feature_names,
                optimized_features=feature_names,
                n_original=len(feature_names),
                n_optimized=len(feature_names),
                improvement=0.0,
                best_score=0.0,
                n_trials=0,
            )

        def objective(trial):
            selected = [i for i, f in enumerate(feature_names)
                        if trial.suggest_categorical(f"f_{i}", [True, False])]

            if len(selected) < min_features:
                return 0.0

            X_train_sub = X_train[:, selected]
            X_val_sub = X_val[:, selected]

            model = ModelRegistry.create(self.model_name)
            model.fit(X_train_sub, y_train, X_val_sub, y_val)
            preds = model.predict(X_val_sub)
            return f1_score(y_val, preds.class_predictions, average="weighted")

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)

        best_features = [f for i, f in enumerate(feature_names)
                         if study.best_trial.params.get(f"f_{i}", False)]

        return OptimizationResult(
            model_name=self.model_name,
            original_features=feature_names,
            optimized_features=best_features,
            n_original=len(feature_names),
            n_optimized=len(best_features),
            improvement=0.0,
            best_score=study.best_value,
            n_trials=self.n_trials,
        )
```

---

## Implementation Checklist

### Task 1.1: FeatureRegistry
- [ ] Create `src/features/registry.py`
- [ ] `FeatureDefinition` dataclass
- [ ] `FeatureRegistry` singleton with `@register` decorator
- [ ] `get_by_family()`, `compute()`, `list_all()` methods

### Task 1.2: Feature Definitions
- [ ] Register all 5 RAW features
- [ ] Register all 23 MOMENTUM features
- [ ] Register all 16 MOVING_AVERAGE features
- [ ] Register all 25 VOLATILITY features
- [ ] Register all 15 VOLUME features
- [ ] Register all 6 TREND features
- [ ] Register all 12 PRICE features
- [ ] Register all 15 MICROSTRUCTURE features
- [ ] Register all 12 ENTROPY features
- [ ] Register all 15 WAVELETS features
- [ ] Register all 9 TEMPORAL features
- [ ] Register all 9 REGIME features
- [ ] Register MTF indicator variants

### Task 1.3: Model Strategies
- [ ] Create `src/features/strategies.py`
- [ ] `ModelFeatureStrategy` dataclass
- [ ] Define strategies for all 22 models
- [ ] `get_strategy_for_model()` function

### Task 1.4: Feature Optimization
- [ ] Create `src/features/optimization.py`
- [ ] `OptimizationResult` dataclass
- [ ] `FeatureOptimizer` class with Optuna
- [ ] Integration with ModelRegistry

### Total Feature Count
| Family | Count |
|--------|-------|
| raw | 5 |
| momentum | 23 |
| moving_average | 16 |
| volatility | 25 |
| volume | 15 |
| trend | 6 |
| price | 12 |
| microstructure | 15 |
| entropy | 12 |
| wavelets | 15 |
| temporal | 9 |
| regime | 9 |
| **TOTAL** | **162** |

**+ MTF**: ~240 features (30 per TF x 8 higher TFs)
**Grand Total**: ~400 features available

---

## Next Phase

After feature computation, proceed to **PHASE_1B: LABELING & OPTIMIZATION** for:
- Triple-barrier label generation with Optuna optimization
- Feature selection optimization with Optuna
- Feature pruning with Optuna
- Hyperparameter optimization with Optuna

---

## Document Metadata

| Field | Value |
|-------|-------|
| Version | 1.0 |
| Created | 2026-01-16 |
| Purpose | Unified feature engineering |
| Related Docs | PHASE_0_FOUNDATION.md, PHASE_1B_LABELING_OPTIMIZATION.md |
