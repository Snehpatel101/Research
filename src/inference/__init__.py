"""
Inference package for ML Model Factory.

This package provides end-to-end inference capabilities:
- ModelBundle: Serializable container for model artifacts
- InferencePipeline: High-level prediction interface
- BatchPredictor: Efficient batch processing
- ModelServer: Optional HTTP serving

Usage:
    # Bundle a trained model
    from src.inference import ModelBundle

    bundle = ModelBundle.from_training(
        model=trained_model,
        scaler=fitted_scaler,
        feature_columns=feature_cols,
        horizon=20,
    )
    bundle.save("./bundles/xgb_h20")

    # Load and predict
    from src.inference import InferencePipeline

    pipeline = InferencePipeline.from_bundle("./bundles/xgb_h20")
    result = pipeline.predict(X_new)

    # Batch inference
    from src.inference import BatchPredictor

    predictor = BatchPredictor.from_bundle("./bundles/xgb_h20")
    result = predictor.predict_batch(df, output_path="predictions.parquet")
"""

from src.inference.batch import (
    BatchPredictor,
    BatchProgress,
    BatchResult,
    run_batch_inference,
)
from src.inference.bundle import (
    BUNDLE_VERSION,
    BundleManifest,
    BundleMetadata,
    ModelBundle,
)
from src.inference.pipeline import (
    EnsembleResult,
    InferencePipeline,
    InferenceResult,
)
from src.inference.server import (
    ModelServer,
    ServerConfig,
    start_server,
)

__all__ = [
    # Bundle
    "ModelBundle",
    "BundleMetadata",
    "BundleManifest",
    "BUNDLE_VERSION",
    # Pipeline
    "InferencePipeline",
    "InferenceResult",
    "EnsembleResult",
    # Batch
    "BatchPredictor",
    "BatchProgress",
    "BatchResult",
    "run_batch_inference",
    # Server
    "ModelServer",
    "ServerConfig",
    "start_server",
]
