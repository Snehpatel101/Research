#!/usr/bin/env bash
# Packaging smoke test: build the sdist and wheel (uv builds the wheel from the
# sdist, so both are exercised), install the wheel -- not editable -- into a
# fresh venv with CPU torch, pinned to uv.lock, and import it from outside the
# source tree. Catches modules or runtime data files (src/config/global.yaml)
# missing from the distribution. Used by CI (`package` job) and
# `make wheel-smoke`.
#
#   scripts/wheel_smoke.sh            # Python 3.11
#   PYTHON=3.12 scripts/wheel_smoke.sh
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

# setuptools re-adds every file listed in a leftover *.egg-info/SOURCES.txt to
# the sdist, so a stale one from an older MANIFEST.in would mask a regression.
rm -rf "$root"/*.egg-info
uv build --out-dir "$work/dist" "$root"

# The sdist ships the package and its metadata files, not the repository.
sdist_files="$(tar -tzf "$work"/dist/*.tar.gz | cut -d/ -f2- | sort)"
for required in README.md LICENSE CHANGELOG.md pyproject.toml src/config/global.yaml; do
    grep -qx "$required" <<<"$sdist_files" || { echo "sdist is missing $required" >&2; exit 1; }
done
if grep -qE '^(tests|scripts|docs|notebooks|examples)/|^CLAUDE\.md$' <<<"$sdist_files"; then
    echo "sdist ships repository-only files:" >&2
    grep -E '^(tests|scripts|docs|notebooks|examples)/|^CLAUDE\.md$' <<<"$sdist_files" | head >&2
    exit 1
fi

bash "$root/scripts/lock_constraints.sh" >"$work/constraints.txt"
uv venv "$work/venv" --python "${PYTHON:-3.11}"
export VIRTUAL_ENV="$work/venv"
uv pip install torch --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match -c "$work/constraints.txt"
uv pip install "$work"/dist/*.whl -c "$work/constraints.txt"

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

BASE_MODELS = {
    "xgboost", "lightgbm", "catboost",  # boosting
    "random_forest", "logistic", "svm",  # classical
    "lstm", "gru",  # RNN
    "tcn", "inceptiontime", "resnet1d",  # CNN
    "transformer", "patchtst", "itransformer", "tft",  # transformer
    "nbeats",  # MLP
}
META_LEARNERS = {"ridge_meta", "xgboost_meta", "mlp_meta", "calibrated_meta", "voting_meta"}
missing = (BASE_MODELS | META_LEARNERS) - set(ModelRegistry.list_all())
assert not missing, f"not registered: {sorted(missing)}"
print(f"wheel imports OK: {src.__file__}, {ModelRegistry.count()} registered models")
EOF
"$work/venv/bin/python" -m src.cli --help >/dev/null
"$work/venv/bin/ensemble-pipeline" --help >/dev/null
echo "wheel smoke OK"
