#!/usr/bin/env bash
# Packaging smoke test: build the sdist and wheel (uv builds the wheel from the
# sdist, so both are exercised), install the wheel -- not editable -- into a
# fresh venv with CPU torch, and import it from outside the source tree.
# Catches modules or runtime data files (src/config/global.yaml) missing from
# the distribution. Used by CI (`package` job) and `make wheel-smoke`.
#
#   scripts/wheel_smoke.sh            # Python 3.11
#   PYTHON=3.12 scripts/wheel_smoke.sh
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

uv build --out-dir "$work/dist" "$root"
uv venv "$work/venv" --python "${PYTHON:-3.11}"
export VIRTUAL_ENV="$work/venv"
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install "$work"/dist/*.whl

cd "$work"  # the source tree must not be importable from here
"$work/venv/bin/python" - <<'EOF'
import importlib
import pkgutil

import src

assert "site-packages" in src.__file__, f"imported from the source tree: {src.__file__}"

# Every module in the wheel imports (a module left out of the wheel breaks its importers).
for mod in pkgutil.walk_packages(src.__path__, "src."):
    importlib.import_module(mod.name)

from src.config.global_config import load_global_config
from src.factory import MLFactory  # noqa: F401
from src.models.registry import ModelRegistry

load_global_config()  # FileNotFoundError if global.yaml was not shipped as package data
assert ModelRegistry.count() >= 16, ModelRegistry.list_all()
print(f"wheel imports OK: {src.__file__}, {ModelRegistry.count()} registered models")
EOF
"$work/venv/bin/python" -m src.cli --help >/dev/null
"$work/venv/bin/ensemble-pipeline" --help >/dev/null
echo "wheel smoke OK"
