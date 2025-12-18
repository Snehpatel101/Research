#!/bin/bash
# Verification script for labeling pipeline installation

echo "=========================================="
echo "Labeling Pipeline - Installation Verification"
echo "=========================================="
echo ""

# Check Python version
echo "1. Checking Python version..."
python3 --version
echo ""

# Check dependencies
echo "2. Checking dependencies..."
python3 << 'PYEOF'
import sys
deps = {
    'numba': 'numba',
    'deap': 'deap', 
    'scipy': 'scipy',
    'matplotlib': 'matplotlib',
    'pandas': 'pandas',
    'numpy': 'numpy',
    'tqdm': 'tqdm'
}

all_ok = True
for name, module in deps.items():
    try:
        mod = __import__(module)
        version = getattr(mod, '__version__', 'unknown')
        print(f"  ✓ {name:12s} {version}")
    except ImportError:
        print(f"  ✗ {name:12s} NOT INSTALLED")
        all_ok = False

if all_ok:
    print("\n  All dependencies installed!")
else:
    print("\n  Some dependencies missing. Run: pip install -r requirements_labeling.txt")
    sys.exit(1)
PYEOF

if [ $? -ne 0 ]; then
    exit 1
fi
echo ""

# Check file structure
echo "3. Checking file structure..."
files=(
    "src/stages/stage4_labeling.py"
    "src/stages/stage5_ga_optimize.py"
    "src/stages/stage6_final_labels.py"
    "src/stages/__init__.py"
    "src/run_labeling_pipeline.py"
    "src/stages/test_stages.py"
    "requirements_labeling.txt"
)

all_files_ok=true
for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✓ $file"
    else
        echo "  ✗ $file (missing)"
        all_files_ok=false
    fi
done
echo ""

if [ "$all_files_ok" = false ]; then
    echo "  Some files are missing!"
    exit 1
fi

# Check directories
echo "4. Checking directories..."
dirs=(
    "data/labels"
    "config/ga_results"
    "results/ga_plots"
)

for dir in "${dirs[@]}"; do
    if [ -d "$dir" ]; then
        echo "  ✓ $dir/"
    else
        echo "  ✗ $dir/ (missing)"
    fi
done
echo ""

# Run unit tests
echo "5. Running unit tests..."
python3 src/stages/test_stages.py
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ VERIFICATION COMPLETE"
    echo "=========================================="
    echo ""
    echo "Ready to run labeling pipeline!"
    echo ""
    echo "Quick start:"
    echo "  python src/run_labeling_pipeline.py"
    echo ""
    echo "See LABELING_QUICKSTART.md for usage examples."
else
    echo ""
    echo "=========================================="
    echo "✗ VERIFICATION FAILED"
    echo "=========================================="
    echo ""
    echo "Unit tests failed. Please check the output above."
    exit 1
fi
