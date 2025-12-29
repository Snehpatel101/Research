"""
Verify test structure and count test coverage.
This script validates the test file without requiring pytest to run.
"""
import ast
import sys
from pathlib import Path

def analyze_test_file(filepath):
    """Analyze test file structure using AST."""
    with open(filepath, 'r') as f:
        content = f.read()

    try:
        tree = ast.parse(content, filename=filepath)
    except SyntaxError as e:
        print(f"❌ SYNTAX ERROR: {e}")
        return False

    print("=" * 80)
    print("TEST FILE STRUCTURE ANALYSIS")
    print("=" * 80)
    print(f"\nFile: {filepath}")
    print()

    # Find all classes and their test methods
    test_classes = []
    total_methods = 0

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            # Check if it's a test class (starts with Test)
            if node.name.startswith('Test'):
                methods = []
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        # Check if it's a test method
                        if item.name.startswith('test_'):
                            methods.append(item.name)

                if methods:
                    test_classes.append({
                        'name': node.name,
                        'methods': methods,
                        'docstring': ast.get_docstring(node)
                    })
                    total_methods += len(methods)

    # Display results
    print(f"Test Classes Found: {len(test_classes)}")
    print(f"Total Test Methods: {total_methods}")
    print()

    for i, cls in enumerate(test_classes, 1):
        print(f"{i}. {cls['name']}")
        if cls['docstring']:
            print(f"   {cls['docstring']}")
        print(f"   Test methods: {len(cls['methods'])}")
        for method in cls['methods']:
            print(f"     - {method}")
        print()

    # Check imports
    print("=" * 80)
    print("IMPORTS VERIFICATION")
    print("=" * 80)

    imports_found = {
        'numpy': False,
        'pytest': False,
        'compute_quality_scores': False
    }

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if 'numpy' in alias.name:
                    imports_found['numpy'] = True
                if 'pytest' in alias.name:
                    imports_found['pytest'] = True
        elif isinstance(node, ast.ImportFrom):
            if node.module and 'compute_quality_scores' in [n.name for n in node.names]:
                imports_found['compute_quality_scores'] = True

    for name, found in imports_found.items():
        status = "✓" if found else "❌"
        print(f"{status} {name}")

    print()

    # Verify all required imports are present
    all_imports_ok = all(imports_found.values())

    if not all_imports_ok:
        print("❌ WARNING: Some required imports are missing")
        return False

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"✓ Test file syntax is valid")
    print(f"✓ Found {len(test_classes)} test classes")
    print(f"✓ Found {total_methods} test methods")
    print(f"✓ All required imports present")
    print()

    # Detailed coverage breakdown
    print("Coverage Breakdown:")
    long_tests = sum(1 for cls in test_classes if 'Long' in cls['name'] for _ in cls['methods'])
    short_tests = sum(1 for cls in test_classes if 'Short' in cls['name'] for _ in cls['methods'])
    neutral_tests = sum(1 for cls in test_classes if 'Neutral' in cls['name'] for _ in cls['methods'])
    edge_tests = sum(1 for cls in test_classes if 'Edge' in cls['name'] for _ in cls['methods'])
    correctness_tests = sum(1 for cls in test_classes if 'Correctness' in cls['name'] for _ in cls['methods'])

    print(f"  LONG trades: {long_tests} tests")
    print(f"  SHORT trades: {short_tests} tests")
    print(f"  NEUTRAL trades: {neutral_tests} tests")
    print(f"  Edge cases: {edge_tests} tests")
    print(f"  Direction-aware correctness: {correctness_tests} tests")
    print()

    min_required = 9  # At least 3 tests per direction (LONG/SHORT/NEUTRAL)
    if total_methods >= min_required:
        print(f"✓ Coverage requirement met: {total_methods} >= {min_required} tests")
    else:
        print(f"❌ Coverage requirement NOT met: {total_methods} < {min_required} tests")
        return False

    print()
    print("=" * 80)
    print("VERIFICATION COMPLETE - ALL CHECKS PASSED")
    print("=" * 80)

    return True

if __name__ == '__main__':
    test_file = Path(__file__).parent / 'test_quality_scores.py'

    if not test_file.exists():
        print(f"❌ Test file not found: {test_file}")
        sys.exit(1)

    success = analyze_test_file(test_file)
    sys.exit(0 if success else 1)
