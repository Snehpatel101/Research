"""
Fast cross-process determinism check (not marked slow, so CI runs it).

Random forest and LightGBM fitted with several threads, and clustered MDA
feature importances, computed in two interpreters with different
PYTHONHASHSEED values, must be bit-identical. Catches regressions of the
thread-order summation and hash-order issues without a full pipeline run
(``tests/e2e/test_determinism_e2e.py`` covers the end-to-end path).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from tests.helpers import REPO_ROOT

N_THREADS = 2  # multi-threaded, within the shared machine's budget


def _run(output: Path, hash_seed: int) -> dict[str, np.ndarray]:
    env = {
        **os.environ,
        "PYTHONHASHSEED": str(hash_seed),
        "OMP_NUM_THREADS": str(N_THREADS),
    }
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "tests.integration.threaded_determinism_driver",
            str(output),
            str(N_THREADS),
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr[-4000:]
    with np.load(output) as data:
        return {k: data[k] for k in data.files}


def test_threaded_fits_and_rankings_identical_across_processes(tmp_path: Path) -> None:
    first = _run(tmp_path / "a.npz", hash_seed=1)
    second = _run(tmp_path / "b.npz", hash_seed=3)
    assert sorted(first) == sorted(second)
    for key in first:
        assert np.array_equal(first[key], second[key]), f"{key} differs between processes"
    # The fixture is informative: the models learned the signal
    assert first["proba__random_forest"].shape == (3000, 3)
    assert first["ranking__clustered"][0] in {"f00", "f01"}
