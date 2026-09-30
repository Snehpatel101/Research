#!/usr/bin/env bash
# Print pip constraints pinned to uv.lock (every extra) for `uv pip install -c`.
#
# CI and CPU dev installs cannot `uv sync` from the lock: it resolves torch from
# PyPI (the CUDA build). They install torch from the PyTorch CPU index instead
# and constrain everything to the locked versions with this file. torch stays
# pinned (`torch==X` matches the `X+cpu` wheel). Pins for the CUDA stack
# (triton, nvidia-*, cuda-*) are kept: a constraint only applies to packages
# that get installed, and xgboost pulls nvidia-nccl-cu12 on Linux.
#
#   scripts/lock_constraints.sh > constraints.txt
#   uv pip install torch --index-url https://download.pytorch.org/whl/cpu -c constraints.txt
#   uv pip install -e ".[dev,stats]" -c constraints.txt
set -euo pipefail

cd "$(dirname "$0")/.."
uv export --frozen --no-hashes --no-emit-project --no-header --no-annotate --all-extras
