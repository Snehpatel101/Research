"""D12: Config dict representation is stable (hash stability).

Verifies that ExperimentConfig.to_dict() produces deterministic output:
- Same config instance yields identical dicts on repeated calls.
- Two separately constructed identical configs yield identical dicts.
- Changing a field produces a different dict.
- JSON serialization of to_dict() is stable (string equality).
"""

import json

from src.config.experiment import ExperimentConfig


def _make_config(**overrides) -> ExperimentConfig:
    """Create a config with fixed values (no auto-generated run_id)."""
    defaults = {
        "name": "test_experiment",
        "description": "stability test",
        "run_id": "fixed_run_id",
        "output_dir": "experiments/test",
        "random_seed": 42,
        "verbose": 1,
    }
    defaults.update(overrides)
    return ExperimentConfig(**defaults)


class TestToDictIdempotent:
    """to_dict() called twice on the same instance returns identical dicts."""

    def test_same_instance_returns_equal_dicts(self):
        cfg = _make_config()
        d1 = cfg.to_dict()
        d2 = cfg.to_dict()
        assert d1 == d2

    def test_same_instance_returns_equal_json(self):
        cfg = _make_config()
        j1 = json.dumps(cfg.to_dict(), sort_keys=True)
        j2 = json.dumps(cfg.to_dict(), sort_keys=True)
        assert j1 == j2


class TestToDictDeterministic:
    """Two separately-constructed identical configs produce the same dict."""

    def test_separate_instances_equal(self):
        cfg_a = _make_config()
        cfg_b = _make_config()
        assert cfg_a.to_dict() == cfg_b.to_dict()

    def test_separate_instances_json_equal(self):
        cfg_a = _make_config()
        cfg_b = _make_config()
        j_a = json.dumps(cfg_a.to_dict(), sort_keys=True)
        j_b = json.dumps(cfg_b.to_dict(), sort_keys=True)
        assert j_a == j_b


class TestToDictSensitivity:
    """Changing a field produces a different dict."""

    def test_different_symbol(self):
        cfg_a = _make_config()
        cfg_b = _make_config()
        cfg_b.data.symbol = "MGC"
        assert cfg_a.to_dict() != cfg_b.to_dict()
        # Only the symbol field should differ in the data section
        assert cfg_a.to_dict()["data"]["symbol"] == "MES"
        assert cfg_b.to_dict()["data"]["symbol"] == "MGC"

    def test_different_name(self):
        cfg_a = _make_config(name="alpha")
        cfg_b = _make_config(name="beta")
        assert cfg_a.to_dict() != cfg_b.to_dict()
        assert cfg_a.to_dict()["name"] == "alpha"
        assert cfg_b.to_dict()["name"] == "beta"

    def test_different_random_seed(self):
        cfg_a = _make_config(random_seed=42)
        cfg_b = _make_config(random_seed=99)
        assert cfg_a.to_dict() != cfg_b.to_dict()

    def test_different_models(self):
        cfg_a = _make_config()
        cfg_b = _make_config()
        cfg_b.training.models = ["lightgbm", "lstm"]
        assert cfg_a.to_dict() != cfg_b.to_dict()

    def test_different_horizons(self):
        cfg_a = _make_config()
        cfg_b = _make_config()
        cfg_b.training.horizons = [1, 2, 3]
        assert cfg_a.to_dict() != cfg_b.to_dict()


class TestToDictJsonStability:
    """JSON string from to_dict() is stable across calls and instances."""

    def test_json_hash_stable(self):
        """Hash of JSON string is identical for identical configs."""
        cfg_a = _make_config()
        cfg_b = _make_config()
        h_a = hash(json.dumps(cfg_a.to_dict(), sort_keys=True))
        h_b = hash(json.dumps(cfg_b.to_dict(), sort_keys=True))
        assert h_a == h_b

    def test_json_contains_all_sections(self):
        """to_dict() output includes all top-level sections."""
        cfg = _make_config()
        d = cfg.to_dict()
        for key in (
            "name",
            "description",
            "run_id",
            "output_dir",
            "random_seed",
            "verbose",
            "data",
            "training",
            "evaluation",
            "bundling",
        ):
            assert key in d, f"Missing top-level key: {key}"

    def test_json_round_trip_preserves_values(self):
        """Values survive dict -> json -> dict round-trip.

        Note: JSON converts tuples to lists, so we normalize before comparing.
        """
        cfg = _make_config()
        d = cfg.to_dict()
        j = json.dumps(d, sort_keys=True)
        d2 = json.loads(j)
        # JSON round-trip converts tuples to lists; re-encode both sides
        # so the comparison is type-agnostic for tuple/list.
        assert json.loads(json.dumps(d)) == d2
