"""
Data Versioning and Manifest Management
Tracks artifacts, checksums, and changes between pipeline runs
"""

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class ArtifactManifest:
    """Manages artifact tracking and versioning for pipeline runs."""

    def __init__(self, run_id: str, project_root: Path):
        """
        Initialize manifest for a pipeline run.

        Args:
            run_id: Unique identifier for this run
            project_root: Root directory of the project
        """
        self.run_id = run_id
        self.project_root = Path(project_root)
        self.run_dir = self.project_root / "runs" / run_id
        self.manifest_path = self.run_dir / "artifacts" / "manifest.json"
        self.artifacts: dict[str, dict[str, Any]] = {}
        self.metadata: dict[str, Any] = {
            "run_id": run_id,
            "created_at": datetime.now().isoformat(),
            "manifest_version": "1.0",
        }

    def compute_file_checksum(self, file_path: Path, algorithm: str = "sha256") -> str:
        """
        Compute checksum for a file.

        Args:
            file_path: Path to file
            algorithm: Hash algorithm (sha256, md5)

        Returns:
            Hexadecimal checksum string
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        hash_func = hashlib.new(algorithm)

        # Handle different file types
        if file_path.suffix == ".parquet":
            # For parquet files, hash the content data
            df = pd.read_parquet(file_path)
            # Create a stable representation
            data_str = df.to_json(orient="records", date_format="iso")
            hash_func.update(data_str.encode())
        else:
            # For regular files, hash the bytes
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hash_func.update(chunk)

        return hash_func.hexdigest()

    def compute_dataframe_checksum(self, df: pd.DataFrame, algorithm: str = "sha256") -> str:
        """
        Compute checksum for a pandas DataFrame.

        Args:
            df: DataFrame to hash
            algorithm: Hash algorithm

        Returns:
            Hexadecimal checksum string
        """
        hash_func = hashlib.new(algorithm)

        # Create stable JSON representation
        data_str = df.to_json(orient="records", date_format="iso")
        hash_func.update(data_str.encode())

        return hash_func.hexdigest()

    def add_artifact(
        self,
        name: str,
        file_path: Path | None = None,
        artifact_type: str = "file",
        stage: str = "unknown",
        metadata: dict[str, Any] | None = None,
        compute_checksum: bool = True,
    ):
        """
        Add an artifact to the manifest.

        Args:
            name: Unique name for the artifact
            file_path: Path to the artifact file
            artifact_type: Type of artifact (file, directory, dataframe)
            stage: Pipeline stage that created this artifact
            metadata: Additional metadata
            compute_checksum: Whether to compute file checksum
        """
        artifact_info: dict[str, Any] = {
            "name": name,
            "type": artifact_type,
            "stage": stage,
            "created_at": datetime.now().isoformat(),
        }

        if file_path:
            file_path = Path(file_path)
            artifact_info["path"] = str(file_path)
            artifact_info["exists"] = file_path.exists()

            if file_path.exists():
                artifact_info["size_bytes"] = file_path.stat().st_size

                if compute_checksum and artifact_type == "file":
                    try:
                        artifact_info["checksum"] = self.compute_file_checksum(file_path)
                        artifact_info["checksum_algorithm"] = "sha256"
                    except Exception as e:
                        logger.warning(f"Failed to compute checksum for {file_path}: {e}")

        if metadata:
            artifact_info["metadata"] = metadata

        self.artifacts[name] = artifact_info
        logger.debug(f"Added artifact: {name}")

    def add_stage_artifacts(
        self, stage: str, files: list[Path], metadata: dict[str, Any] | None = None
    ):
        """
        Add multiple artifacts from a pipeline stage.

        Args:
            stage: Pipeline stage name
            files: List of file paths produced by this stage
            metadata: Additional metadata for the stage
        """
        for file_path in files:
            file_path = Path(file_path)
            artifact_name = f"{stage}_{file_path.name}"
            self.add_artifact(
                name=artifact_name,
                file_path=file_path,
                artifact_type="file",
                stage=stage,
                metadata=metadata,
            )

        logger.info(f"Added {len(files)} artifacts from stage: {stage}")

    def get_artifact(self, name: str) -> dict[str, Any] | None:
        """Get artifact information by name."""
        return self.artifacts.get(name)

    def get_stage_artifacts(self, stage: str) -> dict[str, dict[str, Any]]:
        """Get all artifacts for a specific stage."""
        return {name: info for name, info in self.artifacts.items() if info["stage"] == stage}

    def verify_artifact(self, name: str) -> bool:
        """
        Verify that an artifact exists and checksum matches.

        Args:
            name: Artifact name

        Returns:
            True if artifact is valid, False otherwise
        """
        artifact = self.get_artifact(name)
        if not artifact:
            logger.warning(f"Artifact not found in manifest: {name}")
            return False

        file_path = Path(artifact.get("path", ""))
        if not file_path.exists():
            logger.warning(f"Artifact file missing: {file_path}")
            return False

        if "checksum" in artifact:
            try:
                current_checksum = self.compute_file_checksum(file_path)
                expected_checksum = artifact["checksum"]

                if current_checksum != expected_checksum:
                    logger.warning(
                        f"Checksum mismatch for {name}: "
                        f"expected {expected_checksum}, got {current_checksum}"
                    )
                    return False
            except Exception as e:
                logger.warning(f"Failed to verify checksum for {name}: {e}")
                return False

        return True

    def verify_all_artifacts(self) -> dict[str, bool]:
        """
        Verify all artifacts in the manifest.

        Returns:
            Dictionary mapping artifact names to verification status
        """
        results = {}
        for name in self.artifacts:
            results[name] = self.verify_artifact(name)

        valid_count = sum(results.values())
        total_count = len(results)
        logger.info(f"Verified {valid_count}/{total_count} artifacts")

        return results

    def save(self, path: Path | None = None):
        """
        Save manifest to JSON file.

        Args:
            path: Path to save manifest (defaults to run_dir/artifacts/manifest.json)
        """
        if path is None:
            path = self.manifest_path

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        manifest_data = {
            "metadata": self.metadata,
            "artifacts": self.artifacts,
            "saved_at": datetime.now().isoformat(),
        }

        with open(path, "w") as f:
            json.dump(manifest_data, f, indent=2)

        logger.info(f"Manifest saved to {path}")

    @classmethod
    def load(cls, run_id: str, project_root: Path) -> "ArtifactManifest":
        """
        Load manifest from disk.

        Args:
            run_id: Run identifier
            project_root: Project root directory

        Returns:
            ArtifactManifest instance
        """
        manifest = cls(run_id=run_id, project_root=project_root)

        if not manifest.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest.manifest_path}")

        with open(manifest.manifest_path) as f:
            data = json.load(f)

        manifest.metadata = data.get("metadata", manifest.metadata)
        manifest.artifacts = data.get("artifacts", {})

        logger.info(f"Loaded manifest with {len(manifest.artifacts)} artifacts")
        return manifest

    def compare_with(self, other: "ArtifactManifest") -> dict[str, Any]:
        """
        Compare this manifest with another to find changes.

        Args:
            other: Another ArtifactManifest to compare with

        Returns:
            Dictionary containing added, removed, and modified artifacts
        """
        self_names = set(self.artifacts.keys())
        other_names = set(other.artifacts.keys())

        added = self_names - other_names
        removed = other_names - self_names
        common = self_names & other_names

        modified = []
        for name in common:
            self_artifact = self.artifacts[name]
            other_artifact = other.artifacts[name]

            # Compare checksums if available
            self_checksum = self_artifact.get("checksum")
            other_checksum = other_artifact.get("checksum")

            if self_checksum and other_checksum and self_checksum != other_checksum:
                modified.append(name)

        return {
            "added": sorted(added),
            "removed": sorted(removed),
            "modified": sorted(modified),
            "unchanged": sorted(common - set(modified)),
        }

    def get_summary(self) -> dict[str, Any]:
        """Get summary statistics about the manifest."""
        stages = set(artifact["stage"] for artifact in self.artifacts.values())
        total_size = sum(artifact.get("size_bytes", 0) for artifact in self.artifacts.values())

        return {
            "run_id": self.run_id,
            "total_artifacts": len(self.artifacts),
            "stages": sorted(stages),
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
            "created_at": self.metadata.get("created_at"),
        }

    def print_summary(self):
        """Print a human-readable summary of the manifest."""
        summary = self.get_summary()

        print(f"\nManifest Summary for Run: {summary['run_id']}")
        print("=" * 60)
        print(f"Total Artifacts: {summary['total_artifacts']}")
        print(f"Total Size: {summary['total_size_mb']:.2f} MB")
        print(f"Created: {summary['created_at']}")
        print(f"\nStages: {', '.join(summary['stages'])}")
        print("\nArtifacts by Stage:")

        for stage in summary["stages"]:
            stage_artifacts = self.get_stage_artifacts(stage)
            print(f"  {stage}: {len(stage_artifacts)} artifacts")


def compare_runs(run_id_1: str, run_id_2: str, project_root: Path | None = None) -> dict[str, Any]:
    """
    Compare manifests between two runs.

    Args:
        run_id_1: First run ID
        run_id_2: Second run ID
        project_root: Project root directory

    Returns:
        Comparison results
    """
    if project_root is None:
        project_root = Path(__file__).parent.parent.resolve()

    try:
        manifest_1 = ArtifactManifest.load(run_id_1, project_root)
        manifest_2 = ArtifactManifest.load(run_id_2, project_root)

        comparison = manifest_1.compare_with(manifest_2)

        logger.info(f"Compared runs: {run_id_1} vs {run_id_2}")
        logger.info(f"  Added: {len(comparison['added'])}")
        logger.info(f"  Removed: {len(comparison['removed'])}")
        logger.info(f"  Modified: {len(comparison['modified'])}")
        logger.info(f"  Unchanged: {len(comparison['unchanged'])}")

        return comparison

    except FileNotFoundError as e:
        logger.error(f"Failed to compare runs: {e}")
        raise


if __name__ == "__main__":
    # Example usage - configure logging for standalone execution
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler())

    # Create a test manifest
    manifest = ArtifactManifest(run_id="20241218_120000", project_root=Path("/home/user/Research"))

    # Add some test artifacts
    manifest.add_artifact(
        name="clean_data_MES",
        file_path=Path("/home/user/Research/data/clean/MES_5m_clean.parquet"),
        artifact_type="file",
        stage="cleaning",
        metadata={"symbol": "MES", "rows": 100000},
    )

    manifest.add_artifact(
        name="features_MES",
        file_path=Path("/home/user/Research/data/features/MES_5m_features.parquet"),
        artifact_type="file",
        stage="features",
        metadata={"symbol": "MES", "num_features": 50},
    )

    # Print summary
    manifest.print_summary()

    # Save manifest
    manifest.save()

    # Verify artifacts
    verification = manifest.verify_all_artifacts()
    print(f"\nVerification: {sum(verification.values())}/{len(verification)} valid")
