"""
Model Contract - Declares input requirements for each model type.

Every registered model must declare its requirements so the
adapter system can route data correctly.

Phase 0 of the SNwH (Unified Multi-Timeframe Model Factory) implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from src.core.types import DataRank

from .data_contract import FeatureMode, MTFMode

if TYPE_CHECKING:
    from .data_contract import DataContract


class ModelContractViolation(Exception):
    """Raised when data violates model contract requirements."""

    def __init__(self, model_name: str, issues: list[str]):
        self.model_name = model_name
        self.issues = issues
        super().__init__(f"Model '{model_name}' contract violations: {issues}")


@dataclass(frozen=True)
class ModelContract:
    """
    Input requirements contract for a model.

    Every model declares:
    - What data rank it expects (2D/3D/4D)
    - What feature mode it prefers (engineered/raw/hybrid)
    - What MTF mode it uses (none/indicators/multi_stream)
    - Its primary timeframe preference
    - Sequence length requirements (for 3D/4D)

    This contract is used by:
    1. AdapterRegistry to route data correctly
    2. Trainer to load appropriate timeframe data
    3. Validator to check compatibility before training
    """

    # Model identity
    model_name: str
    model_family: str  # boosting, neural, classical, ensemble, meta_learner

    # Input requirements
    input_rank: DataRank = DataRank.TABULAR_2D
    feature_mode: FeatureMode = FeatureMode.ENGINEERED
    mtf_mode: MTFMode = MTFMode.NONE

    # Timeframe configuration
    primary_timeframe: str = "5min"  # Default primary TF
    mtf_timeframes: tuple[str, ...] = ()  # Additional TFs for multi_stream

    # Sequence requirements (for 3D/4D models)
    sequence_length: int = 60
    patch_length: int | None = None  # For patch-based transformers

    # Scaling requirements
    requires_scaling: bool = True
    scaler_type: str = "robust"  # robust, standard, minmax, none

    # Feature bounds
    min_features: int = 4
    max_features: int = 200

    # Description
    description: str = ""

    @property
    def requires_sequences(self) -> bool:
        """Whether model requires sequential (3D+) input."""
        return self.input_rank.value >= 3

    @property
    def requires_multi_timeframe(self) -> bool:
        """Whether model requires multi-timeframe (4D) input."""
        return self.input_rank == DataRank.MULTI_TF_4D

    @property
    def adapter_id(self) -> str:
        """
        Determine the adapter ID based on requirements.

        Returns:
            Adapter identifier: "tabular", "sequence", or "multi_stream"
        """
        if self.input_rank == DataRank.TABULAR_2D:
            return "tabular"
        elif self.input_rank == DataRank.SEQUENCE_3D:
            return "sequence"
        elif self.input_rank == DataRank.MULTI_TF_4D:
            return "multi_stream"
        else:
            raise ValueError(f"Unknown input rank: {self.input_rank}")

    def validate_data_contract(self, data_contract: DataContract) -> tuple[bool, list[str]]:
        """
        Validate that a data contract satisfies this model's requirements.

        Args:
            data_contract: DataContract to validate

        Returns:
            (is_valid, list_of_issues)
        """
        issues = []

        # Check rank compatibility
        if data_contract.data_rank != self.input_rank:
            issues.append(
                f"Data rank mismatch: model expects {self.input_rank.value}D, "
                f"data is {data_contract.data_rank.value}D"
            )

        # Check sequence length for 3D/4D
        if self.requires_sequences:
            if (
                data_contract.sequence_length
                and data_contract.sequence_length != self.sequence_length
            ):
                issues.append(
                    f"Sequence length mismatch: model expects {self.sequence_length}, "
                    f"data has {data_contract.sequence_length}"
                )

        # Check timeframe count for 4D
        if self.requires_multi_timeframe:
            expected_n_tf = len(self.mtf_timeframes) + 1  # primary + mtf
            if data_contract.n_timeframes and data_contract.n_timeframes != expected_n_tf:
                issues.append(
                    f"Timeframe count mismatch: model expects {expected_n_tf}, "
                    f"data has {data_contract.n_timeframes}"
                )

        # Check feature count bounds
        if data_contract.n_features < self.min_features:
            issues.append(
                f"Too few features: model needs >= {self.min_features}, "
                f"data has {data_contract.n_features}"
            )
        if data_contract.n_features > self.max_features:
            issues.append(
                f"Too many features: model max is {self.max_features}, "
                f"data has {data_contract.n_features}"
            )

        return len(issues) == 0, issues

    def validate_data_contract_strict(self, data_contract: DataContract) -> None:
        """
        Validate data contract, raising ModelContractViolation on failure.

        This is the preferred method for validation in pipelines where
        incompatible data should halt execution.

        Args:
            data_contract: DataContract to validate

        Raises:
            ModelContractViolation: If validation fails with list of issues
        """
        is_valid, issues = self.validate_data_contract(data_contract)
        if not is_valid:
            raise ModelContractViolation(self.model_name, issues)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "model_name": self.model_name,
            "model_family": self.model_family,
            "input_rank": self.input_rank.value,
            "feature_mode": self.feature_mode.value,
            "mtf_mode": self.mtf_mode.value,
            "primary_timeframe": self.primary_timeframe,
            "mtf_timeframes": list(self.mtf_timeframes),
            "sequence_length": self.sequence_length,
            "patch_length": self.patch_length,
            "requires_scaling": self.requires_scaling,
            "scaler_type": self.scaler_type,
            "min_features": self.min_features,
            "max_features": self.max_features,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelContract:
        """Deserialize from dictionary."""
        return cls(
            model_name=data["model_name"],
            model_family=data["model_family"],
            input_rank=DataRank(data.get("input_rank", 2)),
            feature_mode=FeatureMode(data.get("feature_mode", "engineered")),
            mtf_mode=MTFMode(data.get("mtf_mode", "none")),
            primary_timeframe=data.get("primary_timeframe", "5min"),
            mtf_timeframes=tuple(data.get("mtf_timeframes", [])),
            sequence_length=data.get("sequence_length", 60),
            patch_length=data.get("patch_length"),
            requires_scaling=data.get("requires_scaling", True),
            scaler_type=data.get("scaler_type", "robust"),
            min_features=data.get("min_features", 4),
            max_features=data.get("max_features", 200),
            description=data.get("description", ""),
        )

    def __repr__(self) -> str:
        """Concise string representation."""
        return (
            f"ModelContract({self.model_name}, {self.model_family}, "
            f"rank={self.input_rank.value}D, tf={self.primary_timeframe})"
        )


# =============================================================================
# MODEL CONTRACT REGISTRY - All 23 Models
# =============================================================================

MODEL_CONTRACTS: dict[str, ModelContract] = {
    # =========================================================================
    # BOOSTING MODELS (3 models) - 2D tabular, no scaling needed
    # =========================================================================
    "xgboost": ModelContract(
        model_name="xgboost",
        model_family="boosting",
        input_rank=DataRank.TABULAR_2D,
        feature_mode=FeatureMode.ENGINEERED,
        mtf_mode=MTFMode.INDICATORS,
        primary_timeframe="15min",
        requires_scaling=False,
        scaler_type="none",
        min_features=40,
        max_features=200,
        description="XGBoost gradient boosting with GPU support",
    ),
    "lightgbm": ModelContract(
        model_name="lightgbm",
        model_family="boosting",
        input_rank=DataRank.TABULAR_2D,
        feature_mode=FeatureMode.ENGINEERED,
        mtf_mode=MTFMode.INDICATORS,
        primary_timeframe="15min",
        requires_scaling=False,
        scaler_type="none",
        min_features=40,
        max_features=200,
        description="LightGBM gradient boosting with GPU support",
    ),
    "catboost": ModelContract(
        model_name="catboost",
        model_family="boosting",
        input_rank=DataRank.TABULAR_2D,
        feature_mode=FeatureMode.ENGINEERED,
        mtf_mode=MTFMode.INDICATORS,
        primary_timeframe="15min",
        requires_scaling=False,
        scaler_type="none",
        min_features=40,
        max_features=200,
        description="CatBoost gradient boosting (conditional registration)",
    ),
    # =========================================================================
    # CLASSICAL MODELS (3 models) - 2D tabular
    # =========================================================================
    "random_forest": ModelContract(
        model_name="random_forest",
        model_family="classical",
        input_rank=DataRank.TABULAR_2D,
        feature_mode=FeatureMode.ENGINEERED,
        mtf_mode=MTFMode.INDICATORS,
        primary_timeframe="15min",
        requires_scaling=False,
        scaler_type="none",
        min_features=30,
        max_features=150,
        description="Random Forest ensemble classifier",
    ),
    "logistic": ModelContract(
        model_name="logistic",
        model_family="classical",
        input_rank=DataRank.TABULAR_2D,
        feature_mode=FeatureMode.ENGINEERED,
        mtf_mode=MTFMode.NONE,
        primary_timeframe="15min",
        requires_scaling=True,
        scaler_type="standard",
        min_features=15,
        max_features=100,
        description="Logistic Regression classifier",
    ),
    "svm": ModelContract(
        model_name="svm",
        model_family="classical",
        input_rank=DataRank.TABULAR_2D,
        feature_mode=FeatureMode.ENGINEERED,
        mtf_mode=MTFMode.NONE,
        primary_timeframe="15min",
        requires_scaling=True,
        scaler_type="standard",
        min_features=15,
        max_features=80,
        description="Support Vector Machine classifier",
    ),
    # =========================================================================
    # NEURAL/RNN MODELS (5 models) - 3D sequence
    # =========================================================================
    "lstm": ModelContract(
        model_name="lstm",
        model_family="neural",
        input_rank=DataRank.SEQUENCE_3D,
        feature_mode=FeatureMode.ENGINEERED,
        mtf_mode=MTFMode.INDICATORS,
        primary_timeframe="5min",
        sequence_length=60,
        requires_scaling=True,
        scaler_type="robust",
        min_features=50,
        max_features=150,
        description="LSTM recurrent neural network",
    ),
    "gru": ModelContract(
        model_name="gru",
        model_family="neural",
        input_rank=DataRank.SEQUENCE_3D,
        feature_mode=FeatureMode.ENGINEERED,
        mtf_mode=MTFMode.INDICATORS,
        primary_timeframe="5min",
        sequence_length=60,
        requires_scaling=True,
        scaler_type="robust",
        min_features=50,
        max_features=150,
        description="GRU recurrent neural network",
    ),
    "tcn": ModelContract(
        model_name="tcn",
        model_family="neural",
        input_rank=DataRank.SEQUENCE_3D,
        feature_mode=FeatureMode.ENGINEERED,
        mtf_mode=MTFMode.NONE,
        primary_timeframe="5min",
        sequence_length=120,  # TCN benefits from longer sequences
        requires_scaling=True,
        scaler_type="robust",
        min_features=50,
        max_features=120,
        description="Temporal Convolutional Network",
    ),
    "transformer": ModelContract(
        model_name="transformer",
        model_family="neural",
        input_rank=DataRank.SEQUENCE_3D,
        feature_mode=FeatureMode.HYBRID,
        mtf_mode=MTFMode.INDICATORS,
        primary_timeframe="5min",
        sequence_length=128,  # Longer context for attention
        requires_scaling=True,
        scaler_type="standard",
        min_features=20,
        max_features=100,
        description="Vanilla Transformer encoder",
    ),
    "patchtst": ModelContract(
        model_name="patchtst",
        model_family="transformer",
        input_rank=DataRank.MULTI_TF_4D,  # 4D: (N, T, seq_len, features)
        feature_mode=FeatureMode.RAW,  # PatchTST learns from raw data
        mtf_mode=MTFMode.MULTI_STREAM,
        primary_timeframe="1min",
        mtf_timeframes=("5min", "15min"),
        sequence_length=60,
        patch_length=16,
        requires_scaling=True,
        scaler_type="standard",
        min_features=4,  # Raw OHLCV
        max_features=10,
        description="PatchTST Transformer with patching",
    ),
    # =========================================================================
    # TRANSFORMER MODELS (5 models) - 3D sequence
    # =========================================================================
    "itransformer": ModelContract(
        model_name="itransformer",
        model_family="transformer",
        input_rank=DataRank.MULTI_TF_4D,  # 4D: (N, T, seq_len, features)
        feature_mode=FeatureMode.RAW,
        mtf_mode=MTFMode.MULTI_STREAM,
        primary_timeframe="1min",
        mtf_timeframes=("5min", "15min"),
        sequence_length=60,
        requires_scaling=True,
        scaler_type="robust",
        min_features=4,
        max_features=10,
        description="iTransformer with inverted attention",
    ),
    "tft": ModelContract(
        model_name="tft",
        model_family="neural",
        input_rank=DataRank.SEQUENCE_3D,
        feature_mode=FeatureMode.HYBRID,
        mtf_mode=MTFMode.INDICATORS,
        primary_timeframe="5min",
        sequence_length=60,
        requires_scaling=True,
        scaler_type="robust",
        min_features=20,
        max_features=80,
        description="Temporal Fusion Transformer",
    ),
    "nbeats": ModelContract(
        model_name="nbeats",
        model_family="neural",
        input_rank=DataRank.SEQUENCE_3D,
        feature_mode=FeatureMode.RAW,
        mtf_mode=MTFMode.NONE,
        primary_timeframe="5min",
        sequence_length=60,
        requires_scaling=True,
        scaler_type="robust",
        min_features=2,  # Can work with minimal features
        max_features=20,
        description="N-BEATS interpretable time series model",
    ),
    "inceptiontime": ModelContract(
        model_name="inceptiontime",
        model_family="neural",
        input_rank=DataRank.SEQUENCE_3D,
        feature_mode=FeatureMode.ENGINEERED,
        mtf_mode=MTFMode.NONE,
        primary_timeframe="5min",
        sequence_length=60,
        requires_scaling=True,
        scaler_type="robust",
        min_features=30,
        max_features=100,
        description="InceptionTime multi-scale CNN",
    ),
    "resnet1d": ModelContract(
        model_name="resnet1d",
        model_family="neural",
        input_rank=DataRank.SEQUENCE_3D,
        feature_mode=FeatureMode.ENGINEERED,
        mtf_mode=MTFMode.NONE,
        primary_timeframe="5min",
        sequence_length=60,
        requires_scaling=True,
        scaler_type="robust",
        min_features=30,
        max_features=100,
        description="1D ResNet with residual connections",
    ),
    # =========================================================================
    # ENSEMBLE MODELS (3 models) - Dynamic/2D
    # =========================================================================
    "voting": ModelContract(
        model_name="voting",
        model_family="ensemble",
        input_rank=DataRank.TABULAR_2D,  # Meta-input is 2D
        feature_mode=FeatureMode.ENGINEERED,
        mtf_mode=MTFMode.NONE,  # Varies based on bases
        primary_timeframe="5min",  # Varies
        requires_scaling=False,
        scaler_type="none",
        min_features=10,
        max_features=200,
        description="Voting ensemble (hard/soft)",
    ),
    "stacking": ModelContract(
        model_name="stacking",
        model_family="ensemble",
        input_rank=DataRank.TABULAR_2D,  # OOF predictions are 2D
        feature_mode=FeatureMode.OOF_PROBS,
        mtf_mode=MTFMode.NONE,
        primary_timeframe="5min",
        requires_scaling=False,
        scaler_type="none",
        min_features=3,  # min 1 base model * 3 classes
        max_features=60,  # max 20 base models * 3 classes
        description="Stacking ensemble with OOF predictions",
    ),
    "blending": ModelContract(
        model_name="blending",
        model_family="ensemble",
        input_rank=DataRank.TABULAR_2D,  # Holdout predictions are 2D
        feature_mode=FeatureMode.OOF_PROBS,
        mtf_mode=MTFMode.NONE,
        primary_timeframe="5min",
        requires_scaling=False,
        scaler_type="none",
        min_features=3,
        max_features=60,
        description="Blending ensemble with holdout predictions",
    ),
    # =========================================================================
    # META-LEARNER MODELS (4 models) - Always 2D, OOF input
    # =========================================================================
    "ridge_meta": ModelContract(
        model_name="ridge_meta",
        model_family="meta_learner",
        input_rank=DataRank.TABULAR_2D,
        feature_mode=FeatureMode.OOF_PROBS,
        mtf_mode=MTFMode.NONE,
        primary_timeframe="5min",
        requires_scaling=False,  # Ridge handles scaling internally
        scaler_type="none",
        min_features=3,
        max_features=60,
        description="Ridge regression meta-learner",
    ),
    "mlp_meta": ModelContract(
        model_name="mlp_meta",
        model_family="meta_learner",
        input_rank=DataRank.TABULAR_2D,
        feature_mode=FeatureMode.OOF_PROBS,
        mtf_mode=MTFMode.NONE,
        primary_timeframe="5min",
        requires_scaling=True,
        scaler_type="standard",
        min_features=3,
        max_features=60,
        description="MLP neural network meta-learner",
    ),
    "calibrated_meta": ModelContract(
        model_name="calibrated_meta",
        model_family="meta_learner",
        input_rank=DataRank.TABULAR_2D,
        feature_mode=FeatureMode.OOF_PROBS,
        mtf_mode=MTFMode.NONE,
        primary_timeframe="5min",
        requires_scaling=False,
        scaler_type="none",
        min_features=3,
        max_features=60,
        description="Calibrated classifier meta-learner (isotonic/Platt)",
    ),
    "xgboost_meta": ModelContract(
        model_name="xgboost_meta",
        model_family="meta_learner",
        input_rank=DataRank.TABULAR_2D,
        feature_mode=FeatureMode.OOF_PROBS,
        mtf_mode=MTFMode.NONE,
        primary_timeframe="5min",
        requires_scaling=False,
        scaler_type="none",
        min_features=3,
        max_features=60,
        description="XGBoost gradient boosting meta-learner",
    ),
}


def get_model_contract(model_name: str) -> ModelContract:
    """
    Get the contract for a model.

    Args:
        model_name: Model name (e.g., "xgboost", "lstm")

    Returns:
        ModelContract for the model

    Raises:
        ValueError: If model not found
    """
    name_lower = model_name.lower().strip()
    if name_lower not in MODEL_CONTRACTS:
        available = sorted(MODEL_CONTRACTS.keys())
        raise ValueError(f"No contract for model '{model_name}'. Available: {available}")
    return MODEL_CONTRACTS[name_lower]


def list_model_contracts(family: str | None = None) -> dict[str, ModelContract]:
    """
    List all model contracts, optionally filtered by family.

    Args:
        family: Optional family filter (boosting, neural, classical, ensemble, meta_learner)

    Returns:
        Dictionary of model_name -> ModelContract
    """
    if family is None:
        return MODEL_CONTRACTS.copy()

    family_lower = family.lower().strip()
    return {
        name: contract
        for name, contract in MODEL_CONTRACTS.items()
        if contract.model_family == family_lower
    }


def get_models_by_rank(rank: DataRank) -> list[str]:
    """Get all model names that use a specific data rank."""
    return [name for name, contract in MODEL_CONTRACTS.items() if contract.input_rank == rank]


def get_models_requiring_scaling() -> list[str]:
    """Get all model names that require feature scaling."""
    return [name for name, contract in MODEL_CONTRACTS.items() if contract.requires_scaling]


def get_models_by_mtf_mode(mtf_mode: MTFMode) -> list[str]:
    """Get all model names that use a specific MTF mode."""
    return [name for name, contract in MODEL_CONTRACTS.items() if contract.mtf_mode == mtf_mode]


__all__ = [
    "ModelContract",
    "ModelContractViolation",
    "MODEL_CONTRACTS",
    "get_model_contract",
    "list_model_contracts",
    "get_models_by_rank",
    "get_models_requiring_scaling",
    "get_models_by_mtf_mode",
]
