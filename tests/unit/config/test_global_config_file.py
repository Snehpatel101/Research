"""The process-wide defaults file ships inside the package.

``load_global_config()`` used to read ``<repo>/config/global.yaml``, a path that
does not exist in a wheel install: every ``get_config_value`` call then logged a
warning and fell back to its hard-coded default. The file now lives next to the
loader as package data (the CI ``package`` job checks it is in the wheel).
"""

from __future__ import annotations

from pathlib import Path

import yaml

import src.config
from src.config.global_config import GlobalConfig, load_global_config


def test_default_global_config_is_package_data() -> None:
    package_file = Path(src.config.__file__).parent / "global.yaml"
    assert package_file.is_file()

    loaded = load_global_config()

    assert loaded.to_dict() == GlobalConfig.from_yaml(package_file).to_dict()


def test_global_config_values_come_from_the_file_not_fallbacks() -> None:
    raw = yaml.safe_load((Path(src.config.__file__).parent / "global.yaml").read_text())

    loaded = load_global_config()

    assert loaded.random_seed == raw["random_seed"]
    assert loaded.training.batch_size == raw["training"]["batch_size"]
    assert loaded.horizons.active == raw["horizons"]["active"]
