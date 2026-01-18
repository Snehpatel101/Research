"""
MetaLearnerFactory - Factory for creating meta-learners from PipelineConfig.

Single entry point for meta-learner creation with config-driven parameters.

PHASE_4: Meta-Learner Factory Integration

This module provides a unified factory pattern for instantiating meta-learners
used in stacking ensembles. It integrates with PipelineConfig for seamless
configuration-driven creation.

Supported Meta-Learners:
- ridge_meta: L2-regularized linear meta-learner (fast, interpretable)
- mlp_meta: 2-layer neural network (captures non-linear interactions)
- xgboost_meta: XGBoost with optional calibration (strong for diverse bases)
- calibrated_meta: Isotonic/Platt calibration wrapper (probability calibration)

Usage:
    from src.core import PipelineConfig
    from src.models.ensemble import MetaLearnerFactory

    config = PipelineConfig(
        symbol="MES",
        data_path="./data/mes.parquet",
        output_dir="./out",
        meta_learner="xgboost_meta",
    )

    factory = MetaLearnerFactory(config)
    meta_learner = factory.create()

    # Or with custom parameters
    meta_learner = factory.create(
        n_estimators=200,
        max_depth=4,
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type

if TYPE_CHECKING:
    from src.core import PipelineConfig

logger = logging.getLogger(__name__)


@dataclass
class MetaLearnerConfig:
    """
    Configuration dataclass for meta-learner parameters.

    This provides a structured way to define meta-learner configurations
    with sensible defaults for each type.

    Attributes:
        name: Meta-learner name (ridge_meta, mlp_meta, xgboost_meta, calibrated_meta)
        alphas: Ridge alpha values for cross-validation
        hidden_layers: MLP hidden layer sizes
        dropout: MLP dropout rate
        learning_rate: MLP learning rate
        max_iter: MLP maximum iterations
        n_estimators: XGBoost number of estimators
        max_depth: XGBoost tree depth
        xgb_learning_rate: XGBoost learning rate
        calibrate: Whether to calibrate XGBoost predictions
        method: Calibration method (isotonic or sigmoid)
        cv: Cross-validation folds for calibration
    """

    name: str
    # Ridge parameters
    alphas: List[float] = field(default_factory=lambda: [0.1, 1.0, 10.0, 100.0])
    # MLP parameters
    hidden_layers: tuple = (64, 32)
    dropout: float = 0.2
    learning_rate: float = 0.001
    max_iter: int = 500
    # XGBoost parameters
    n_estimators: int = 100
    max_depth: int = 3
    xgb_learning_rate: float = 0.1
    calibrate: bool = True
    # Calibration parameters
    method: str = "isotonic"
    cv: int = 3

    def to_dict(self, meta_learner_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Convert config to dictionary for specific meta-learner.

        Args:
            meta_learner_name: Override the name for filtering parameters

        Returns:
            Dictionary of parameters relevant to the meta-learner
        """
        name = meta_learner_name or self.name

        if name == "ridge_meta":
            return {"alphas": self.alphas}
        elif name == "mlp_meta":
            return {
                "hidden_layer_sizes": self.hidden_layers,
                "alpha": self.dropout,  # Used as L2 regularization
                "learning_rate_init": self.learning_rate,
                "max_iter": self.max_iter,
                "early_stopping": True,
            }
        elif name == "xgboost_meta":
            return {
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "learning_rate": self.xgb_learning_rate,
            }
        elif name == "calibrated_meta":
            return {
                "method": self.method,
                "cv": self.cv,
            }
        else:
            return {}


# Meta-learner registry (populated on first access)
META_LEARNER_REGISTRY: Dict[str, Type] = {}

# Flag to track initialization
_REGISTRY_INITIALIZED = False


def register_meta_learner(name: str):
    """
    Decorator to register a meta-learner class.

    Args:
        name: Registry name for the meta-learner

    Returns:
        Decorator function

    Example:
        @register_meta_learner("custom_meta")
        class CustomMetaLearner(BaseModel):
            ...
    """

    def decorator(cls):
        META_LEARNER_REGISTRY[name] = cls
        logger.debug(f"Registered meta-learner: {name}")
        return cls

    return decorator


def _load_meta_learners() -> None:
    """
    Load all meta-learner classes into registry.

    This function is called lazily on first access to avoid
    circular import issues.
    """
    global _REGISTRY_INITIALIZED

    if _REGISTRY_INITIALIZED:
        return

    try:
        from src.models.ensemble.calibrated_meta import CalibratedMetaLearner
        from src.models.ensemble.mlp_meta import MLPMetaLearner
        from src.models.ensemble.ridge_meta import RidgeMetaLearner
        from src.models.ensemble.xgboost_meta import XGBoostMeta

        META_LEARNER_REGISTRY.update(
            {
                "ridge_meta": RidgeMetaLearner,
                "mlp_meta": MLPMetaLearner,
                "xgboost_meta": XGBoostMeta,
                "calibrated_meta": CalibratedMetaLearner,
            }
        )
        _REGISTRY_INITIALIZED = True
        logger.debug(f"Loaded {len(META_LEARNER_REGISTRY)} meta-learners into registry")

    except ImportError as e:
        logger.warning(f"Failed to load some meta-learners: {e}")
        _REGISTRY_INITIALIZED = True


def _ensure_registry_loaded() -> None:
    """Ensure the registry is loaded before access."""
    if not _REGISTRY_INITIALIZED:
        _load_meta_learners()


class MetaLearnerFactory:
    """
    Factory for creating meta-learners from PipelineConfig.

    Provides single entry point for meta-learner instantiation with
    config-driven parameters.

    The factory supports:
    - Creating meta-learners from PipelineConfig.meta_learner
    - Custom parameter overrides via kwargs
    - Listing available meta-learners
    - Getting meta-learner information and descriptions

    Attributes:
        config: Optional PipelineConfig instance
        _default_meta_learner: Default meta-learner name from config

    Usage:
        from src.core import PipelineConfig
        from src.models.ensemble import MetaLearnerFactory

        config = PipelineConfig(
            symbol="MES",
            data_path="./data/mes.parquet",
            output_dir="./out",
            meta_learner="xgboost_meta",
        )

        factory = MetaLearnerFactory(config)
        meta_learner = factory.create()

        # Or with custom parameters
        meta_learner = factory.create(
            n_estimators=200,
            max_depth=4,
        )

        # Create different meta-learner than config default
        ridge_meta = factory.create("ridge_meta", alpha=0.5)
    """

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        """
        Initialize factory.

        Args:
            config: Optional PipelineConfig. If provided, uses config.meta_learner
                   as the default meta-learner name.
        """
        _ensure_registry_loaded()
        self.config = config
        self._default_meta_learner = config.meta_learner if config else "ridge_meta"

    @classmethod
    def from_config(cls, config: PipelineConfig) -> MetaLearnerFactory:
        """
        Create factory from PipelineConfig.

        Args:
            config: PipelineConfig instance

        Returns:
            MetaLearnerFactory instance
        """
        return cls(config)

    def create(
        self,
        meta_learner_name: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Create meta-learner instance.

        Args:
            meta_learner_name: Meta-learner name. If None, uses config.meta_learner.
            **kwargs: Additional parameters to pass to meta-learner constructor.
                     These override the default parameters.

        Returns:
            Meta-learner instance (RidgeMetaLearner, MLPMetaLearner,
            XGBoostMeta, or CalibratedMetaLearner)

        Raises:
            ValueError: If meta_learner_name is not in registry

        Example:
            factory = MetaLearnerFactory(config)

            # Use config default
            meta = factory.create()

            # Override with specific meta-learner
            meta = factory.create("mlp_meta", hidden_layer_sizes=(128, 64))

            # Use config default with parameter overrides
            meta = factory.create(n_estimators=200)
        """
        _ensure_registry_loaded()

        name = meta_learner_name or self._default_meta_learner

        if name not in META_LEARNER_REGISTRY:
            raise ValueError(
                f"Unknown meta-learner: {name}. "
                f"Available: {list(META_LEARNER_REGISTRY.keys())}"
            )

        meta_class = META_LEARNER_REGISTRY[name]

        # Build config dict
        config_dict = self._build_config_dict(name, **kwargs)

        logger.info(f"Creating meta-learner: {name}")

        return meta_class(config=config_dict)

    def _build_config_dict(self, name: str, **kwargs: Any) -> Dict[str, Any]:
        """
        Build configuration dictionary for meta-learner.

        Args:
            name: Meta-learner name
            **kwargs: Override parameters

        Returns:
            Configuration dictionary
        """
        # Start with defaults based on meta-learner type
        defaults = self._get_defaults(name)

        # Override with kwargs
        defaults.update(kwargs)

        return defaults

    def _get_defaults(self, name: str) -> Dict[str, Any]:
        """
        Get default parameters for a meta-learner.

        Each meta-learner has carefully tuned defaults suitable for
        stacking ensemble use cases.

        Args:
            name: Meta-learner name

        Returns:
            Dictionary of default parameters
        """
        if name == "ridge_meta":
            return {
                "alpha": 1.0,
                "fit_intercept": True,
                "class_weight": "balanced",
                "scale_features": True,
            }
        elif name == "mlp_meta":
            return {
                "hidden_layer_sizes": (64, 32),
                "alpha": 0.01,  # L2 regularization
                "learning_rate_init": 0.001,
                "max_iter": 500,
                "early_stopping": True,
                "validation_fraction": 0.1,
                "n_iter_no_change": 10,
                "activation": "relu",
                "solver": "adam",
                "scale_features": True,
            }
        elif name == "xgboost_meta":
            return {
                "n_estimators": 100,
                "max_depth": 3,
                "learning_rate": 0.1,
                "min_child_weight": 10,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "gamma": 0.1,
                "reg_alpha": 0.1,
                "reg_lambda": 1.0,
                "early_stopping_rounds": 20,
            }
        elif name == "calibrated_meta":
            return {
                "base_estimator": "logistic",
                "method": "isotonic",
                "cv": 5,
                "ensemble": True,
                "scale_features": True,
            }
        else:
            return {}

    @staticmethod
    def list_available() -> List[str]:
        """
        List available meta-learners.

        Returns:
            List of registered meta-learner names
        """
        _ensure_registry_loaded()
        return list(META_LEARNER_REGISTRY.keys())

    @staticmethod
    def get_meta_learner_info(name: str) -> Dict[str, Any]:
        """
        Get information about a meta-learner.

        Args:
            name: Meta-learner name

        Returns:
            Dictionary with name, class, description, and default parameters

        Raises:
            ValueError: If meta-learner name is not registered
        """
        _ensure_registry_loaded()

        if name not in META_LEARNER_REGISTRY:
            raise ValueError(f"Unknown meta-learner: {name}")

        meta_class = META_LEARNER_REGISTRY[name]

        # Descriptions for each meta-learner
        descriptions = {
            "ridge_meta": (
                "L2-regularized linear meta-learner. "
                "Fast training with closed-form solution. "
                "Interpretable weights show relative model contribution. "
                "Best for well-calibrated base models."
            ),
            "mlp_meta": (
                "2-layer neural network meta-learner. "
                "Captures non-linear interactions between base model predictions. "
                "Automatic feature learning from prediction patterns. "
                "Best when base models have complementary error patterns."
            ),
            "xgboost_meta": (
                "XGBoost gradient boosting meta-learner. "
                "Captures complex non-linear interactions with built-in regularization. "
                "Feature importance for model contribution analysis. "
                "Best when base models are diverse."
            ),
            "calibrated_meta": (
                "Calibration wrapper meta-learner using Isotonic or Platt scaling. "
                "Ensures predicted probabilities reflect true class frequencies. "
                "Essential for threshold-based trading decisions. "
                "Best for probability calibration in ensemble outputs."
            ),
        }

        # Default parameters for each meta-learner
        default_params = {
            "ridge_meta": {
                "alpha": 1.0,
                "class_weight": "balanced",
            },
            "mlp_meta": {
                "hidden_layer_sizes": (64, 32),
                "alpha": 0.01,
                "max_iter": 500,
            },
            "xgboost_meta": {
                "n_estimators": 100,
                "max_depth": 3,
                "learning_rate": 0.1,
            },
            "calibrated_meta": {
                "method": "isotonic",
                "cv": 5,
                "base_estimator": "logistic",
            },
        }

        return {
            "name": name,
            "class": meta_class.__name__,
            "module": meta_class.__module__,
            "description": descriptions.get(name, "No description available"),
            "default_params": default_params.get(name, {}),
        }

    @staticmethod
    def get_all_info() -> Dict[str, Dict[str, Any]]:
        """
        Get information about all available meta-learners.

        Returns:
            Dictionary mapping meta-learner names to their info
        """
        _ensure_registry_loaded()
        return {
            name: MetaLearnerFactory.get_meta_learner_info(name)
            for name in META_LEARNER_REGISTRY
        }


def get_meta_learner(
    name: str,
    config: Optional[PipelineConfig] = None,
    **kwargs: Any,
) -> Any:
    """
    Convenience function to get a meta-learner by name.

    This is a shorthand for creating a factory and calling create().

    Args:
        name: Meta-learner name
        config: Optional PipelineConfig
        **kwargs: Additional parameters

    Returns:
        Meta-learner instance

    Example:
        # Simple usage
        meta = get_meta_learner("ridge_meta")

        # With parameters
        meta = get_meta_learner("xgboost_meta", n_estimators=200)

        # With config
        meta = get_meta_learner("mlp_meta", config=pipeline_config)
    """
    factory = MetaLearnerFactory(config)
    return factory.create(name, **kwargs)


def create_meta_learner_from_config(config: PipelineConfig) -> Any:
    """
    Create meta-learner from PipelineConfig.

    Uses config.meta_learner to determine which meta-learner to create.
    This is the recommended way to create meta-learners in the pipeline.

    Args:
        config: PipelineConfig instance

    Returns:
        Meta-learner instance matching config.meta_learner

    Example:
        config = PipelineConfig(
            symbol="MES",
            data_path="./data/mes.parquet",
            output_dir="./out",
            meta_learner="xgboost_meta",
        )

        meta = create_meta_learner_from_config(config)
        # Returns XGBoostMeta instance
    """
    factory = MetaLearnerFactory(config)
    return factory.create()


def list_meta_learners() -> List[str]:
    """
    List all available meta-learner names.

    Convenience function that wraps MetaLearnerFactory.list_available().

    Returns:
        List of meta-learner names
    """
    return MetaLearnerFactory.list_available()


__all__ = [
    # Main factory
    "MetaLearnerFactory",
    # Configuration
    "MetaLearnerConfig",
    # Registry
    "META_LEARNER_REGISTRY",
    "register_meta_learner",
    # Convenience functions
    "get_meta_learner",
    "create_meta_learner_from_config",
    "list_meta_learners",
]
