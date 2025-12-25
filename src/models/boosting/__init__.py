"""
Boosting model implementations.

Tree-based gradient boosting models that work on tabular features
without requiring scaling or sequential input.

Models:
- XGBoost: Fast, regularized gradient boosting with GPU support
- LightGBM: Leaf-wise growth, handles large datasets efficiently
- CatBoost: Excellent categorical feature handling with ordered boosting

All models auto-register with ModelRegistry on import.

Example:
    # Models register automatically when imported
    from src.models.boosting import XGBoostModel, LightGBMModel, CatBoostModel

    # Or create via registry
    from src.models import ModelRegistry
    model = ModelRegistry.create("xgboost", config={"max_depth": 8})
    model = ModelRegistry.create("lgbm", config={"num_leaves": 63})
    model = ModelRegistry.create("catboost", config={"depth": 8})
"""
from .xgboost_model import XGBoostModel
from .lightgbm_model import LightGBMModel
from .catboost_model import CatBoostModel

__all__ = [
    "XGBoostModel",
    "LightGBMModel",
    "CatBoostModel",
]
