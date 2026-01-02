"""
Boosting model implementations.

Tree-based gradient boosting models that work on tabular features
without requiring scaling or sequential input.

Models:
- XGBoost: Fast, regularized gradient boosting with GPU support
- LightGBM: Leaf-wise growth, handles large datasets efficiently
- CatBoost: Excellent categorical feature handling with ordered boosting (optional)

All models auto-register with ModelRegistry on import if their dependencies
are available. CatBoost is optional and only registered when installed.

Example:
    # Models register automatically when imported
    from src.models.boosting import XGBoostModel, LightGBMModel

    # CatBoost is optional - check availability first
    from src.models.boosting.catboost_model import CATBOOST_AVAILABLE
    if CATBOOST_AVAILABLE:
        from src.models.boosting import CatBoostModel

    # Or create via registry (only registered models appear)
    from src.models import ModelRegistry
    model = ModelRegistry.create("xgboost", config={"max_depth": 8})
    model = ModelRegistry.create("lgbm", config={"num_leaves": 63})
    # catboost only available if installed
"""

from .catboost_model import CATBOOST_AVAILABLE, CatBoostModel
from .lightgbm_model import LightGBMModel
from .xgboost_model import XGBoostModel

__all__ = [
    "XGBoostModel",
    "LightGBMModel",
    "CatBoostModel",
    "CATBOOST_AVAILABLE",
]
