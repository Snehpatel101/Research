#!/usr/bin/env python3
"""
Test script for the pipeline configuration system
Demonstrates all major features without running the full pipeline
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pipeline_config import PipelineConfig, create_default_config
from manifest import ArtifactManifest


def test_config_creation():
    """Test configuration creation and validation."""
    print("=" * 70)
    print("TEST 1: Configuration Creation and Validation")
    print("=" * 70)

    # Create config
    config = create_default_config(
        symbols=['MES', 'MGC', 'MNQ'],
        start_date='2020-01-01',
        end_date='2024-12-31',
        run_id='test_run_001',
        description='Test configuration system'
    )

    print("\n✓ Configuration created successfully")
    print(f"  Run ID: {config.run_id}")
    print(f"  Symbols: {', '.join(config.symbols)}")
    print(f"  Date Range: {config.start_date} to {config.end_date}")

    # Validate
    issues = config.validate()
    if issues:
        print("\n✗ Validation failed:")
        for issue in issues:
            print(f"    - {issue}")
        return False
    else:
        print("\n✓ Configuration validated successfully")
        return True


def test_config_persistence():
    """Test saving and loading configuration."""
    print("\n" + "=" * 70)
    print("TEST 2: Configuration Persistence")
    print("=" * 70)

    # Create config
    config = create_default_config(
        symbols=['MES', 'MGC'],
        run_id='test_run_002',
        description='Test save/load'
    )

    # Save
    config.create_directories()
    config_path = config.save_config()
    print(f"\n✓ Configuration saved to: {config_path}")

    # Load
    loaded_config = PipelineConfig.load_config(config_path)
    print(f"✓ Configuration loaded successfully")
    print(f"  Run ID: {loaded_config.run_id}")
    print(f"  Symbols: {', '.join(loaded_config.symbols)}")

    # Verify
    assert config.run_id == loaded_config.run_id
    assert config.symbols == loaded_config.symbols
    assert config.train_ratio == loaded_config.train_ratio

    print("✓ All values match!")
    return True


def test_config_summary():
    """Test configuration summary generation."""
    print("\n" + "=" * 70)
    print("TEST 3: Configuration Summary")
    print("=" * 70)

    config = create_default_config(
        symbols=['MES', 'MGC'],
        run_id='test_run_003'
    )

    summary = config.summary()
    print(summary)

    return True


def test_manifest():
    """Test manifest creation and artifact tracking."""
    print("\n" + "=" * 70)
    print("TEST 4: Manifest and Artifact Tracking")
    print("=" * 70)

    # Create manifest
    manifest = ArtifactManifest(
        run_id='test_run_004',
        project_root=Path('/home/user/Research')
    )

    print("\n✓ Manifest created")

    # Add mock artifacts
    manifest.add_artifact(
        name='test_artifact_1',
        artifact_type='file',
        stage='testing',
        metadata={'test': True},
        compute_checksum=False
    )

    manifest.add_artifact(
        name='test_artifact_2',
        artifact_type='file',
        stage='testing',
        metadata={'count': 100},
        compute_checksum=False
    )

    print(f"✓ Added {len(manifest.artifacts)} artifacts")

    # Get summary
    summary = manifest.get_summary()
    print(f"\nManifest Summary:")
    print(f"  Run ID: {summary['run_id']}")
    print(f"  Total Artifacts: {summary['total_artifacts']}")
    print(f"  Stages: {', '.join(summary['stages'])}")

    return True


def test_config_validation_errors():
    """Test that validation catches errors."""
    print("\n" + "=" * 70)
    print("TEST 5: Configuration Validation (Error Detection)")
    print("=" * 70)

    # Create config with invalid ratios
    try:
        config = PipelineConfig(
            run_id='test_run_005',
            symbols=['MES'],
            train_ratio=0.5,
            val_ratio=0.3,
            test_ratio=0.3  # Total = 1.1 (invalid!)
        )
        print("\n✗ Should have raised ValueError for invalid ratios")
        return False
    except ValueError as e:
        print(f"\n✓ Correctly caught invalid ratios: {e}")

    # Create config with other invalid parameters
    config = PipelineConfig(
        run_id='test_run_005',
        symbols=['MES'],
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        barrier_k_up=-1.0,  # Invalid!
        label_horizons=[-1, 0, 5]  # Invalid!
    )

    issues = config.validate()
    if issues:
        print(f"✓ Found {len(issues)} validation issues:")
        for issue in issues:
            print(f"    - {issue}")
        return True
    else:
        print("✗ Should have found validation issues")
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("PIPELINE CONFIGURATION SYSTEM TEST SUITE")
    print("=" * 70)

    tests = [
        ("Configuration Creation", test_config_creation),
        ("Configuration Persistence", test_config_persistence),
        ("Configuration Summary", test_config_summary),
        ("Manifest Tracking", test_manifest),
        ("Validation Errors", test_config_validation_errors),
    ]

    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n✗ Test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {name}")

    passed = sum(1 for _, success in results if success)
    total = len(results)

    print("\n" + "=" * 70)
    print(f"TOTAL: {passed}/{total} tests passed")
    print("=" * 70)

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
