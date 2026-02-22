"""
Deploy artifact packaging for ML Factory.

Provides a pure-JSON manifest that indexes all bundles for a given
horizon, enabling single-call loading for production inference:

    artifact = load_deploy_artifact("./deploy", horizon=20)
    pred = artifact.predict_from_raw(raw_bars_df)

The deploy directory structure:
    deploy/
        manifest.json           # DeployManifest (lists all horizons + bundles)
        bundles/
            xgboost_h20/        # ModelBundle directories
            lstm_h20/
            ensemble_h20/       # EnsembleBundle directory
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEPLOY_MANIFEST_FILE = "manifest.json"
DEPLOY_VERSION = "1.0.0"


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class HorizonArtifactEntry:
    """Entry for a single model bundle within a horizon."""

    model_name: str
    bundle_path: str  # Relative to deploy dir
    is_ensemble: bool = False
    metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HorizonArtifactEntry:
        return cls(
            model_name=data["model_name"],
            bundle_path=data["bundle_path"],
            is_ensemble=data.get("is_ensemble", False),
            metrics=data.get("metrics", {}),
        )


@dataclass
class HorizonManifest:
    """Manifest for all bundles at a single horizon."""

    horizon: int
    entries: list[HorizonArtifactEntry] = field(default_factory=list)
    primary_model: str = ""  # Name of the recommended model for this horizon

    def to_dict(self) -> dict[str, Any]:
        return {
            "horizon": self.horizon,
            "entries": [e.to_dict() for e in self.entries],
            "primary_model": self.primary_model,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HorizonManifest:
        return cls(
            horizon=data["horizon"],
            entries=[HorizonArtifactEntry.from_dict(e) for e in data.get("entries", [])],
            primary_model=data.get("primary_model", ""),
        )


@dataclass
class DeployManifest:
    """Top-level deploy manifest indexing all horizons and bundles.

    Pure JSON — no pickle, no binary dependencies.
    """

    version: str = DEPLOY_VERSION
    created_at: str = ""
    symbol: str = ""
    horizons: dict[int, HorizonManifest] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "created_at": self.created_at,
            "symbol": self.symbol,
            "horizons": {str(h): m.to_dict() for h, m in self.horizons.items()},
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DeployManifest:
        horizons: dict[int, HorizonManifest] = {}
        for h_str, h_data in data.get("horizons", {}).items():
            horizons[int(h_str)] = HorizonManifest.from_dict(h_data)
        return cls(
            version=data.get("version", DEPLOY_VERSION),
            created_at=data.get("created_at", ""),
            symbol=data.get("symbol", ""),
            horizons=horizons,
            extra=data.get("extra", {}),
        )

    def save(self, path: str | Path) -> Path:
        """Save manifest to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved deploy manifest to {path}")
        return path

    @classmethod
    def load(cls, path: str | Path) -> DeployManifest:
        """Load manifest from JSON file."""
        path = Path(path)
        with open(path) as f:
            return cls.from_dict(json.load(f))


# =============================================================================
# PUBLIC API
# =============================================================================


def select_deploy_artifact(
    deploy_dir: str | Path,
    horizon: int,
    model_name: str | None = None,
) -> Path:
    """Select a bundle path from the deploy manifest.

    Args:
        deploy_dir: Path to deploy directory containing manifest.json.
        horizon: Prediction horizon to load.
        model_name: Specific model name. If None, uses the primary model
            for that horizon (ensemble if available, else best single model).

    Returns:
        Absolute path to the selected bundle directory.

    Raises:
        FileNotFoundError: If deploy_dir or manifest doesn't exist.
        KeyError: If horizon or model not found in manifest.
    """
    deploy_dir = Path(deploy_dir)
    manifest_path = deploy_dir / DEPLOY_MANIFEST_FILE

    if not manifest_path.exists():
        raise FileNotFoundError(f"Deploy manifest not found at {manifest_path}")

    manifest = DeployManifest.load(manifest_path)

    if horizon not in manifest.horizons:
        available = sorted(manifest.horizons.keys())
        raise KeyError(f"Horizon {horizon} not in manifest. Available: {available}")

    h_manifest = manifest.horizons[horizon]

    if model_name is None:
        # Use primary model (ensemble preferred)
        if h_manifest.primary_model:
            model_name = h_manifest.primary_model
        elif h_manifest.entries:
            # Fall back to first entry
            model_name = h_manifest.entries[0].model_name
        else:
            raise KeyError(f"No models available for horizon {horizon}")

    # Find matching entry
    for entry in h_manifest.entries:
        if entry.model_name == model_name:
            bundle_path = deploy_dir / entry.bundle_path
            if not bundle_path.exists():
                # Bundle paths may be relative to output_dir (deploy_dir's parent)
                bundle_path = deploy_dir.parent / entry.bundle_path
            if not bundle_path.exists():
                raise FileNotFoundError(
                    f"Bundle directory not found: {bundle_path} "
                    f"(referenced by manifest entry '{model_name}')"
                )
            return bundle_path.resolve()

    available_models = [e.model_name for e in h_manifest.entries]
    raise KeyError(
        f"Model '{model_name}' not found for horizon {horizon}. " f"Available: {available_models}"
    )


def validate_deploy_artifact(deploy_dir: str | Path) -> dict[str, Any]:
    """Validate a deploy artifact directory.

    Checks:
    - manifest.json exists and parses
    - All referenced bundle paths exist
    - Each bundle can be loaded (optional deep check)

    Args:
        deploy_dir: Path to deploy directory.

    Returns:
        Dict with 'valid' (bool) and 'issues' (list[str]).
    """
    deploy_dir = Path(deploy_dir)
    issues: list[str] = []

    manifest_path = deploy_dir / DEPLOY_MANIFEST_FILE
    if not manifest_path.exists():
        return {"valid": False, "issues": [f"Missing {DEPLOY_MANIFEST_FILE}"]}

    try:
        manifest = DeployManifest.load(manifest_path)
    except Exception as e:
        return {"valid": False, "issues": [f"Manifest parse error: {e}"]}

    for horizon, h_manifest in manifest.horizons.items():
        for entry in h_manifest.entries:
            bundle_path = deploy_dir / entry.bundle_path
            if not bundle_path.exists():
                # Bundle paths may be relative to output_dir (deploy_dir's parent)
                bundle_path = deploy_dir.parent / entry.bundle_path
            if not bundle_path.exists():
                issues.append(
                    f"H{horizon}/{entry.model_name}: bundle not found at {entry.bundle_path}"
                )

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "n_horizons": len(manifest.horizons),
        "version": manifest.version,
    }


def load_deploy_artifact(
    deploy_dir: str | Path,
    horizon: int,
    model_name: str | None = None,
) -> Any:
    """Load a deploy artifact bundle for prediction.

    This is the primary entry point for production inference:

        artifact = load_deploy_artifact("./deploy", horizon=20)
        pred = artifact.predict_from_raw(raw_bars_df)

    Args:
        deploy_dir: Path to deploy directory.
        horizon: Prediction horizon.
        model_name: Specific model (None = primary/ensemble).

    Returns:
        ModelBundle or EnsembleBundle instance ready for prediction.

    Raises:
        FileNotFoundError: If deploy dir or bundle not found.
        KeyError: If horizon or model not in manifest.
    """
    bundle_path = select_deploy_artifact(deploy_dir, horizon, model_name)

    # Determine bundle type by checking for ensemble metadata
    ensemble_metadata = bundle_path / "metadata.json"
    if ensemble_metadata.exists():
        with open(ensemble_metadata) as f:
            meta = json.load(f)

        if "meta_learner_name" in meta:
            from src.inference.ensemble_bundle import EnsembleBundle

            logger.info(f"Loading ensemble bundle from {bundle_path}")
            return EnsembleBundle.load(bundle_path)

    from src.inference.bundle import ModelBundle

    logger.info(f"Loading model bundle from {bundle_path}")
    return ModelBundle.load(bundle_path)


__all__ = [
    "DEPLOY_MANIFEST_FILE",
    "DEPLOY_VERSION",
    "DeployManifest",
    "HorizonArtifactEntry",
    "HorizonManifest",
    "load_deploy_artifact",
    "select_deploy_artifact",
    "validate_deploy_artifact",
]
