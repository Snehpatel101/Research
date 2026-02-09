#!/usr/bin/env python3
import os
import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def check_embargo_bars():
    print("--- Checking for hardcoded embargo_bars=10 ---")
    # Search for 'embargo_bars=10' in python files
    cmd = ["grep", "-r", "embargo_bars=10", str(PROJECT_ROOT / "src"), "--include=*.py"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        print("FAIL: Found hardcoded embargo_bars=10:")
        print(result.stdout)
    else:
        print("PASS: No hardcoded embargo_bars=10 found.")

def check_purged_kfold():
    print("\n--- Checking for PurgedKFold in optimization/ ---")
    opt_dir = PROJECT_ROOT / "src" / "optimization"
    if not opt_dir.exists():
        print(f"SKIP: {opt_dir} not found.")
        return

    # Check if files in optimization/ import PurgedKFold
    cmd = ["grep", "-l", "PurgedKFold", "-r", str(opt_dir), "--include=*.py"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    found_files = result.stdout.strip().split('\n')
    
    py_files = list(opt_dir.glob("**/*.py"))
    py_files = [f for f in py_files if f.name != "__init__.py"]
    
    if len(found_files) >= len(py_files) * 0.5: # Simple heuristic: most should have it
        print(f"PASS: PurgedKFold found in {len(found_files)} files.")
    else:
        print(f"WARNING: PurgedKFold only found in {len(found_files)} out of {len(py_files)} files.")

def check_stacking_unsafe():
    print("\n--- Checking for use_default_configs_for_oof=False in stacking.py ---")
    stacking_file = PROJECT_ROOT / "src" / "models" / "ensemble" / "stacking.py"
    if not stacking_file.exists():
        # Try alternate path
        stacking_file = PROJECT_ROOT / "src" / "ml" / "stacking.py"
    
    if not stacking_file.exists():
         # Fallback search
         cmd = ["find", str(PROJECT_ROOT / "src"), "-name", "stacking.py"]
         res = subprocess.run(cmd, capture_output=True, text=True)
         if res.stdout.strip():
             stacking_file = Path(res.stdout.strip().split('\n')[0])

    if stacking_file and stacking_file.exists():
        with open(stacking_file, 'r') as f:
            content = f.read()
            if "use_default_configs_for_oof=False" in content:
                print(f"FAIL: {stacking_file} contains unsafe OOF config!")
            else:
                print(f"PASS: {stacking_file} looks safe (no use_default_configs_for_oof=False found).")
    else:
        print("SKIP: stacking.py not found.")

if __name__ == "__main__":
    check_embargo_bars()
    check_purged_kfold()
    check_stacking_unsafe()
