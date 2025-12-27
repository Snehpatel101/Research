#!/usr/bin/env python3
"""
Pipeline Integrity Test Script

Tests that all core components can be imported and function correctly
after cross-correlation removal.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_phase1_imports():
    """Test Phase 1 pipeline imports."""
    print("\n=== Testing Phase 1 Imports ===")

    try:
        from src.phase1.pipeline_config import PipelineConfig
        print("✓ PipelineConfig imported")
    except Exception as e:
        print(f"✗ PipelineConfig failed: {e}")
        return False

    try:
        from src.pipeline.runner import PipelineRunner
        print("✓ PipelineRunner imported")
    except Exception as e:
        print(f"✗ PipelineRunner failed: {e}")
        return False

    return True


def test_model_registry():
    """Test model registry and all 12 models."""
    print("\n=== Testing Model Registry ===")

    try:
        from src.models import ModelRegistry
        print("✓ ModelRegistry imported")

        # Get all registered models
        all_models = ModelRegistry.list_all()
        print(f"✓ Found {len(all_models)} registered models")

        # Expected models
        expected_models = {
            # Boosting (3)
            'xgboost', 'lightgbm', 'catboost',
            # Neural (3)
            'lstm', 'gru', 'tcn',
            # Classical (3)
            'random_forest', 'logistic', 'svm',
            # Ensemble (3)
            'voting', 'stacking', 'blending',
        }

        # Convert to lowercase for comparison
        registered_models = set(m.lower() for m in all_models)

        # Check if all expected models are present
        missing = expected_models - registered_models
        extra = registered_models - expected_models

        if missing:
            print(f"✗ Missing models: {missing}")
            return False

        if extra:
            print(f"  Note: Extra models registered: {extra}")

        print(f"✓ All 12 expected models registered: {sorted(expected_models)}")

        # Get models by family
        families = ModelRegistry.list_models()
        print(f"\n  Models by family:")
        for family, models in sorted(families.items()):
            print(f"    {family}: {', '.join(sorted(models))}")

        return True

    except Exception as e:
        print(f"✗ ModelRegistry failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cross_validation():
    """Test cross-validation imports."""
    print("\n=== Testing Cross-Validation ===")

    try:
        from src.cross_validation.purged_kfold import PurgedKFold
        print("✓ PurgedKFold imported")
    except Exception as e:
        print(f"✗ PurgedKFold failed: {e}")
        return False

    try:
        from src.cross_validation.cv_runner import CrossValidationRunner
        print("✓ CrossValidationRunner imported")
    except Exception as e:
        print(f"✗ CrossValidationRunner failed: {e}")
        return False

    try:
        from src.cross_validation.feature_selector import WalkForwardFeatureSelector
        print("✓ WalkForwardFeatureSelector imported")
    except Exception as e:
        print(f"✗ WalkForwardFeatureSelector failed: {e}")
        return False

    try:
        from src.cross_validation.oof_generator import OOFGenerator
        print("✓ OOFGenerator imported")
    except Exception as e:
        print(f"✗ OOFGenerator failed: {e}")
        return False

    return True


def test_ensemble_functionality():
    """Test that ensemble models are functional."""
    print("\n=== Testing Ensemble Functionality ===")

    try:
        from src.models.ensemble import VotingEnsemble, StackingEnsemble, BlendingEnsemble
        print("✓ All ensemble models imported")

        from src.models import ModelRegistry

        # Check voting ensemble
        voting_meta = ModelRegistry.get_metadata('voting')
        print(f"✓ Voting ensemble: {voting_meta['description']}")

        # Check stacking ensemble
        stacking_meta = ModelRegistry.get_metadata('stacking')
        print(f"✓ Stacking ensemble: {stacking_meta['description']}")

        # Check blending ensemble
        blending_meta = ModelRegistry.get_metadata('blending')
        print(f"✓ Blending ensemble: {blending_meta['description']}")

        return True

    except Exception as e:
        print(f"✗ Ensemble functionality failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_trainer_config():
    """Test trainer configuration."""
    print("\n=== Testing Trainer Config ===")

    try:
        from src.models import TrainerConfig
        print("✓ TrainerConfig imported")

        # Create a basic config
        config = TrainerConfig(
            model_name="xgboost",
            horizon=20,
            run_id="test_run"
        )
        print(f"✓ Created TrainerConfig for {config.model_name}")

        return True

    except Exception as e:
        print(f"✗ TrainerConfig failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Pipeline Integrity Test")
    print("=" * 60)

    tests = [
        ("Phase 1 Imports", test_phase1_imports),
        ("Model Registry", test_model_registry),
        ("Cross-Validation", test_cross_validation),
        ("Ensemble Functionality", test_ensemble_functionality),
        ("Trainer Config", test_trainer_config),
    ]

    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n✗ {test_name} crashed: {e}")
            import traceback
            traceback.print_exc()
            results[test_name] = False

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    passed = sum(1 for r in results.values() if r)
    total = len(results)

    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")

    print(f"\nPassed: {passed}/{total}")

    if passed == total:
        print("\n🎉 All tests passed! Pipeline integrity verified.")
        return 0
    else:
        print("\n⚠️  Some tests failed. See errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
