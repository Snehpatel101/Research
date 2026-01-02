"""
Classical ML model implementations.

Traditional machine learning models that serve as interpretable
baselines and can be used as meta-learners in ensembles.

Models:
- Random Forest: Ensemble of decision trees
- Logistic Regression: Linear classifier
- SVM: Support Vector Machine

All models auto-register with ModelRegistry on import.

Example:
    # Models register automatically when imported
    from src.models.classical import RandomForestModel

    # Or import all classical models
    from src.models import classical

    # Create via registry
    from src.models import ModelRegistry
    model = ModelRegistry.create("random_forest")
"""

from .logistic import LogisticModel
from .random_forest import RandomForestModel
from .svm import SVMModel

__all__ = [
    "RandomForestModel",
    "LogisticModel",
    "SVMModel",
]
