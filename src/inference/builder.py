"""
BundleBuilder - Create inference bundles from training results.

Bridges PHASE_3 training and PHASE_4 ensemble with PHASE_5 inference.
Uses PipelineConfig as the single source of truth.

This module provides:
- BundleBuilder: Main class for building inference bundles
- BundleBuildResult: Result container with bundle paths and metadata
- build_bundles: Convenience function for quick bundle creation

Example:
    from src.core import PipelineConfig
    from src.inference import BundleBuilder

    config = PipelineConfig(...)
    builder = BundleBuilder(config)

    # Build from PHASE_3 training result
    result = builder.build_from_training_result(training_result)

    # Build from PHASE_4 ensemble result
    result = builder.build_ensemble_bundle(ensemble_result)

    # Or build all at once
    result = builder.build_all(
        training_result=training_result,
        ensemble_result=ensemble_result,
    )
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.core import PipelineConfig

if TYPE_CHECKING:
    from src.models.ensemble.orchestrator import EnsembleResult
    from src.models.training.unified_orchestrator import TrainingRunResult

logger = logging.getLogger(__name__)


# =============================================================================
# RESULT DATACLASS
# =============================================================================


@dataclass
class BundleBuildResult:
    """
    Result from bundle building.

    Attributes:
        bundle_paths: List of paths to created base model bundles
        ensemble_bundle_path: Optional path to ensemble bundle
        n_bundles: Total number of bundles created
        total_size_mb: Total size of all bundles in MB
        metadata: Additional metadata about the build
    """

    bundle_paths: list[Path] = field(default_factory=list)
    ensemble_bundle_path: Path | None = None
    n_bundles: int = 0
    total_size_mb: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "bundle_paths": [str(p) for p in self.bundle_paths],
            "ensemble_bundle_path": (
                str(self.ensemble_bundle_path) if self.ensemble_bundle_path else None
            ),
            "n_bundles": self.n_bundles,
            "total_size_mb": self.total_size_mb,
            "metadata": self.metadata,
        }

    def save(self, path: str | Path) -> Path:
        """
        Save build result to JSON.

        Args:
            path: Path to save the result

        Returns:
            Path to saved file
        """
        path = Path(path)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path

    @classmethod
    def load(cls, path: str | Path) -> BundleBuildResult:
        """
        Load build result from JSON.

        Args:
            path: Path to the saved result

        Returns:
            Loaded BundleBuildResult
        """
        path = Path(path)
        with open(path) as f:
            data = json.load(f)

        return cls(
            bundle_paths=[Path(p) for p in data.get("bundle_paths", [])],
            ensemble_bundle_path=(
                Path(data["ensemble_bundle_path"]) if data.get("ensemble_bundle_path") else None
            ),
            n_bundles=data.get("n_bundles", 0),
            total_size_mb=data.get("total_size_mb", 0.0),
            metadata=data.get("metadata", {}),
        )

    def summary(self) -> str:
        """Get human-readable summary."""
        lines = [
            "BundleBuildResult:",
            f"  Base bundles: {len(self.bundle_paths)}",
            f"  Ensemble bundle: {'Yes' if self.ensemble_bundle_path else 'No'}",
            f"  Total bundles: {self.n_bundles}",
            f"  Total size: {self.total_size_mb:.2f} MB",
        ]
        if self.bundle_paths:
            lines.append("  Bundle paths:")
            for p in self.bundle_paths[:5]:
                lines.append(f"    - {p.name}")
            if len(self.bundle_paths) > 5:
                lines.append(f"    ... and {len(self.bundle_paths) - 5} more")
        return "\n".join(lines)


# =============================================================================
# BUNDLE BUILDER
# =============================================================================


class BundleBuilder:
    """
    Create inference bundles from training results.

    Integrates PHASE_3 training output with PHASE_5 inference bundles.
    Uses PipelineConfig as the single source of truth.

    The builder handles:
    - Extracting trained models from TrainingRunResult
    - Creating PreprocessingGraph for train/serve parity
    - Building ModelBundle for each trained model
    - Building EnsembleBundle from EnsembleResult
    - Computing and storing bundle metadata

    Usage:
        from src.core import PipelineConfig
        from src.inference import BundleBuilder

        config = PipelineConfig(...)
        builder = BundleBuilder(config)

        # Build from PHASE_3 training result
        result = builder.build_from_training_result(training_result)

        # Build from PHASE_4 ensemble result
        result = builder.build_ensemble_bundle(ensemble_result)

        # Or build all at once
        result = builder.build_all(
            training_result=training_result,
            ensemble_result=ensemble_result,
        )
    """

    def __init__(self, config: PipelineConfig) -> None:
        """
        Initialize BundleBuilder.

        Args:
            config: PipelineConfig instance - THE single source of truth
        """
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.bundles_dir = self.output_dir / "bundles"
        self.bundles_dir.mkdir(parents=True, exist_ok=True)

        logger.info("BundleBuilder initialized")
        logger.info(f"  Output: {self.bundles_dir}")

    @classmethod
    def from_config(cls, config: PipelineConfig) -> BundleBuilder:
        """
        Create builder from PipelineConfig.

        Args:
            config: PipelineConfig instance

        Returns:
            BundleBuilder instance
        """
        return cls(config)

    @classmethod
    def from_training_run(cls, run_path: str | Path) -> BundleBuilder:
        """
        Create builder from a completed training run.

        Loads PipelineConfig from the run directory.

        Args:
            run_path: Path to training run directory containing config.json

        Returns:
            BundleBuilder instance

        Raises:
            FileNotFoundError: If config.json not found in run_path
        """
        run_path = Path(run_path)
        config_path = run_path / "config.json"

        if not config_path.exists():
            raise FileNotFoundError(
                f"Config not found at {config_path}. "
                "Ensure the training run saved its configuration."
            )

        config = PipelineConfig.load(config_path)
        return cls(config)

    def build_from_training_result(
        self,
        training_result: TrainingRunResult,
        include_preprocessing_graph: bool = True,
        include_calibrator: bool = True,
        feature_specs: dict[str, Any] | None = None,
    ) -> BundleBuildResult:
        """
        Build bundles from PHASE_3 TrainingRunResult.

        Creates one bundle per trained model, each containing:
        - The trained model
        - Feature scaler
        - Feature column names
        - Preprocessing graph (optional, for raw OHLCV inference)
        - Probability calibrator (optional)
        - Feature spec (optional, for 5-dimension optimization parity)
        - Training metrics

        Args:
            training_result: Result from UnifiedTrainingOrchestrator.train()
            include_preprocessing_graph: Include preprocessing graph for raw inference
            include_calibrator: Include probability calibrator if available
            feature_specs: Optional dict mapping model_key -> FeatureSpec for each model.
                          When provided, the bundle will include the FeatureSpec to ensure
                          inference uses the exact same configuration as training.

        Returns:
            BundleBuildResult with paths to created bundles
        """
        from src.inference.bundle import ModelBundle
        from src.inference.preprocessing_graph import PreprocessingGraph

        bundle_paths: list[Path] = []
        build_metadata: dict[str, Any] = {
            "run_id": training_result.run_id,
            "models_bundled": [],
            "models_skipped": [],
        }

        # Create preprocessing graph if needed
        preprocessing_graph: PreprocessingGraph | None = None
        if include_preprocessing_graph:
            preprocessing_graph = self._create_preprocessing_graph()

        # Build bundle for each model
        for key, model_result in training_result.model_results.items():
            model_name = model_result.model_name
            horizon = model_result.horizon

            logger.info(f"Building bundle for {key}...")

            # Get trainer and extract components
            trainer = model_result.trainer
            if trainer is None:
                logger.warning(f"No trainer for {key}, skipping bundle")
                build_metadata["models_skipped"].append(key)
                continue

            # Extract model - try multiple attribute names for compatibility
            model = self._extract_model(trainer)
            if model is None:
                logger.warning(f"No model in trainer for {key}, skipping")
                build_metadata["models_skipped"].append(key)
                continue

            # Extract scaler
            scaler = self._extract_scaler(trainer)

            # Extract feature columns
            feature_columns = self._extract_feature_columns(trainer, model_result.n_features)

            # Extract calibrator if requested
            calibrator = None
            if include_calibrator:
                calibrator = self._extract_calibrator(trainer)

            # Get feature spec if provided
            feature_spec = None
            if feature_specs is not None:
                feature_spec = feature_specs.get(key)

            # Create bundle
            try:
                bundle = ModelBundle.from_training(
                    model=model,
                    scaler=scaler,
                    feature_columns=feature_columns,
                    horizon=horizon,
                    calibrator=calibrator,
                    preprocessing_graph=preprocessing_graph,
                    feature_spec=feature_spec,
                    symbol=self.config.symbol,
                    training_metrics=model_result.metrics,
                    extra_metadata={
                        "training_run_id": training_result.run_id,
                        "training_time_seconds": model_result.training_time_seconds,
                        "data_rank": model_result.data_rank,
                    },
                )

                # Save bundle
                bundle_path = self.bundles_dir / f"{model_name}_h{horizon}"
                bundle.save(bundle_path, overwrite=True)
                bundle_paths.append(bundle_path)

                build_metadata["models_bundled"].append(
                    {
                        "key": key,
                        "model_name": model_name,
                        "horizon": horizon,
                        "bundle_path": str(bundle_path),
                    }
                )

                logger.info(f"Built bundle: {bundle_path}")

            except Exception as e:
                logger.error(f"Failed to build bundle for {key}: {e}")
                build_metadata["models_skipped"].append(key)
                continue

        # Calculate total size
        total_size = self._calculate_total_size(bundle_paths)

        return BundleBuildResult(
            bundle_paths=bundle_paths,
            n_bundles=len(bundle_paths),
            total_size_mb=total_size,
            metadata=build_metadata,
        )

    def build_ensemble_bundle(
        self,
        ensemble_result: EnsembleResult,
        base_bundles: list[Path] | None = None,
    ) -> Path:
        """
        Build ensemble bundle from PHASE_4 EnsembleResult.

        Creates a bundle containing:
        - The trained meta-learner
        - References to base model bundles
        - Stacking configuration
        - Ensemble metadata

        Args:
            ensemble_result: Result from EnsembleOrchestrator.train()
            base_bundles: Optional paths to base model bundles

        Returns:
            Path to ensemble bundle

        Raises:
            ValueError: If ensemble_result is invalid
        """
        ensemble_dir = self.bundles_dir / "ensemble"
        ensemble_dir.mkdir(parents=True, exist_ok=True)

        # Save ensemble metadata
        ensemble_metadata = {
            "ensemble_name": ensemble_result.ensemble_name,
            "meta_learner_name": ensemble_result.meta_learner_name,
            "base_model_names": ensemble_result.base_model_names,
            "n_base_models": ensemble_result.n_base_models,
            "coverage": ensemble_result.coverage,
            "alignment_offset": ensemble_result.alignment_offset,
            "metrics": ensemble_result.metrics,
            "training_time_seconds": ensemble_result.training_time_seconds,
            "base_bundle_paths": [str(p) for p in (base_bundles or [])],
            "symbol": self.config.symbol,
        }

        metadata_path = ensemble_dir / "ensemble_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(ensemble_metadata, f, indent=2)

        # Save stacking dataset if available
        if ensemble_result.stacking_dataset is not None:
            stacking_path = ensemble_dir / "stacking_dataset.parquet"
            try:
                ensemble_result.stacking_dataset.data.to_parquet(stacking_path)
                logger.info(f"Saved stacking dataset to {stacking_path}")
            except Exception as e:
                logger.warning(f"Failed to save stacking dataset: {e}")

        # Save aligned OOF info if available
        if ensemble_result.aligned_oof is not None:
            aligned_info = {
                "model_names": ensemble_result.aligned_oof.model_names,
                "n_common": ensemble_result.aligned_oof.n_common,
                "coverage": ensemble_result.aligned_oof.coverage,
            }
            aligned_path = ensemble_dir / "aligned_oof_info.json"
            with open(aligned_path, "w") as f:
                json.dump(aligned_info, f, indent=2)

        logger.info(f"Built ensemble bundle: {ensemble_dir}")

        return ensemble_dir

    def build_all(
        self,
        training_result: TrainingRunResult | None = None,
        ensemble_result: EnsembleResult | None = None,
        include_preprocessing_graph: bool = True,
        feature_specs: dict[str, Any] | None = None,
    ) -> BundleBuildResult:
        """
        Build all bundles from training and ensemble results.

        Convenience method that combines build_from_training_result and
        build_ensemble_bundle into a single call.

        Args:
            training_result: PHASE_3 training result (optional)
            ensemble_result: PHASE_4 ensemble result (optional)
            include_preprocessing_graph: Include preprocessing graph
            feature_specs: Optional dict mapping model_key -> FeatureSpec

        Returns:
            BundleBuildResult with all bundle paths

        Raises:
            ValueError: If both training_result and ensemble_result are None
        """
        if training_result is None and ensemble_result is None:
            raise ValueError("At least one of training_result or ensemble_result must be provided")

        bundle_paths: list[Path] = []
        ensemble_path: Path | None = None
        metadata: dict[str, Any] = {}

        # Build base model bundles
        if training_result is not None:
            base_result = self.build_from_training_result(
                training_result,
                include_preprocessing_graph=include_preprocessing_graph,
                feature_specs=feature_specs,
            )
            bundle_paths.extend(base_result.bundle_paths)
            metadata["base_build"] = base_result.metadata

        # Build ensemble bundle
        if ensemble_result is not None:
            ensemble_path = self.build_ensemble_bundle(
                ensemble_result,
                base_bundles=bundle_paths,
            )
            metadata["ensemble_build"] = {
                "ensemble_name": ensemble_result.ensemble_name,
                "n_base_models": ensemble_result.n_base_models,
            }

        # Calculate total size
        all_paths = bundle_paths + ([ensemble_path] if ensemble_path else [])
        total_size = self._calculate_total_size(all_paths)

        # Save build result
        build_result = BundleBuildResult(
            bundle_paths=bundle_paths,
            ensemble_bundle_path=ensemble_path,
            n_bundles=len(bundle_paths) + (1 if ensemble_path else 0),
            total_size_mb=total_size,
            metadata=metadata,
        )

        # Save build result to disk
        result_path = self.bundles_dir / "build_result.json"
        build_result.save(result_path)
        logger.info(f"Saved build result to {result_path}")

        return build_result

    def _create_preprocessing_graph(self) -> Any:
        """
        Create preprocessing graph from config.

        Returns:
            PreprocessingGraph instance configured for the pipeline
        """
        from src.inference.preprocessing_graph import PreprocessingGraph

        # Build pipeline config dict from PipelineConfig
        pipeline_config = {
            "horizons": self.config.horizons,
            "mtf_timeframes": self.config.mtf_timeframes,
            "clean": {
                "source_timeframe": "1min",
                "target_timeframe": "5min",
            },
            "features": {
                "scale_periods": True,
                "base_timeframe": "5min",
            },
            "mtf": {
                "enable_mtf": True,
                "base_timeframe": "5min",
                "mtf_timeframes": self.config.mtf_timeframes,
                "mode": "both",
            },
            "wavelets": {
                "enable_wavelets": True,
            },
            "regime": {
                "enabled": True,
            },
            "scaling": {
                "scaler_type": "robust",
                "clip_outliers": True,
            },
        }

        return PreprocessingGraph.from_pipeline_config(
            pipeline_config=pipeline_config,
            symbol=self.config.symbol,
            horizon=self.config.horizons[0] if self.config.horizons else 20,
        )

    def _extract_model(self, trainer: Any) -> Any | None:
        """
        Extract model from trainer instance.

        Tries multiple attribute names for compatibility with different
        trainer implementations.

        Args:
            trainer: Trainer instance

        Returns:
            Model instance or None
        """
        # Try common attribute names
        for attr in ["model", "_model", "estimator", "_estimator"]:
            model = getattr(trainer, attr, None)
            if model is not None:
                return model

        # Try callable model property
        if hasattr(trainer, "get_model") and callable(trainer.get_model):
            return trainer.get_model()

        return None

    def _extract_scaler(self, trainer: Any) -> Any | None:
        """
        Extract scaler from trainer instance.

        Args:
            trainer: Trainer instance

        Returns:
            Scaler instance or None
        """
        for attr in ["scaler", "_scaler", "feature_scaler", "_feature_scaler"]:
            scaler = getattr(trainer, attr, None)
            if scaler is not None:
                return scaler
        return None

    def _extract_feature_columns(
        self,
        trainer: Any,
        n_features: int,
    ) -> list[str]:
        """
        Extract feature column names from trainer.

        Args:
            trainer: Trainer instance
            n_features: Number of features (fallback for generic names)

        Returns:
            List of feature column names
        """
        # Try common attribute names
        for attr in ["feature_columns", "_feature_columns", "feature_names", "_feature_names"]:
            columns = getattr(trainer, attr, None)
            if columns is not None and len(columns) > 0:
                return list(columns)

        # Try to get from scaler if available
        scaler = self._extract_scaler(trainer)
        if scaler is not None:
            columns = getattr(scaler, "feature_names_in_", None)
            if columns is not None:
                return list(columns)

        # Fallback to generic names
        logger.warning("No feature columns found, using generic names")
        return [f"f{i}" for i in range(n_features)]

    def _extract_calibrator(self, trainer: Any) -> Any | None:
        """
        Extract probability calibrator from trainer.

        Args:
            trainer: Trainer instance

        Returns:
            Calibrator instance or None
        """
        for attr in ["calibrator", "_calibrator", "prob_calibrator"]:
            calibrator = getattr(trainer, attr, None)
            if calibrator is not None:
                return calibrator
        return None

    def _calculate_total_size(self, paths: list[Path]) -> float:
        """
        Calculate total size of bundles in MB.

        Args:
            paths: List of bundle paths (directories)

        Returns:
            Total size in megabytes
        """
        total_bytes = 0
        for path in paths:
            if path.exists():
                if path.is_dir():
                    total_bytes += sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
                else:
                    total_bytes += path.stat().st_size
        return total_bytes / (1024 * 1024)

    def validate_bundles(self) -> dict[str, Any]:
        """
        Validate all bundles in the bundles directory.

        Returns:
            Dictionary with validation results for each bundle
        """
        from src.inference.bundle import ModelBundle

        results: dict[str, Any] = {
            "valid": True,
            "bundles": {},
            "issues": [],
        }

        # Find all bundle directories
        if not self.bundles_dir.exists():
            results["valid"] = False
            results["issues"].append("Bundles directory does not exist")
            return results

        for bundle_dir in self.bundles_dir.iterdir():
            if not bundle_dir.is_dir():
                continue
            if bundle_dir.name == "ensemble":
                # Handle ensemble bundle separately
                continue

            try:
                bundle = ModelBundle.load(bundle_dir)
                validation = bundle.validate()
                results["bundles"][bundle_dir.name] = validation

                if not validation["valid"]:
                    results["valid"] = False
                    for issue in validation.get("issues", []):
                        results["issues"].append(f"{bundle_dir.name}: {issue}")

            except Exception as e:
                results["valid"] = False
                results["bundles"][bundle_dir.name] = {
                    "valid": False,
                    "error": str(e),
                }
                results["issues"].append(f"{bundle_dir.name}: Failed to load - {e}")

        return results


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def build_bundles(
    config: PipelineConfig,
    training_result: TrainingRunResult,
    ensemble_result: EnsembleResult | None = None,
    feature_specs: dict[str, Any] | None = None,
) -> BundleBuildResult:
    """
    Convenience function to build all bundles.

    Creates a BundleBuilder and builds bundles from the provided results.

    Usage:
        from src.core import PipelineConfig
        from src.inference.builder import build_bundles

        result = build_bundles(config, training_result, ensemble_result)
        print(f"Created {result.n_bundles} bundles")

    Args:
        config: PipelineConfig instance
        training_result: PHASE_3 training result
        ensemble_result: Optional PHASE_4 ensemble result
        feature_specs: Optional dict mapping model_key -> FeatureSpec

    Returns:
        BundleBuildResult with all bundle paths
    """
    builder = BundleBuilder(config)
    return builder.build_all(training_result, ensemble_result, feature_specs=feature_specs)


def build_from_run(
    run_path: str | Path,
    training_result: TrainingRunResult,
    ensemble_result: EnsembleResult | None = None,
    feature_specs: dict[str, Any] | None = None,
) -> BundleBuildResult:
    """
    Build bundles from a completed training run.

    Loads config from the run directory and builds bundles.

    Args:
        run_path: Path to training run directory
        training_result: PHASE_3 training result
        ensemble_result: Optional PHASE_4 ensemble result
        feature_specs: Optional dict mapping model_key -> FeatureSpec

    Returns:
        BundleBuildResult with all bundle paths
    """
    builder = BundleBuilder.from_training_run(run_path)
    return builder.build_all(training_result, ensemble_result, feature_specs=feature_specs)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "BundleBuilder",
    "BundleBuildResult",
    "build_bundles",
    "build_from_run",
]
