"""Property 7: ExperimentConfig serialization is lossless for every valid configuration.

``from_dict(to_dict(c))`` (and the YAML path used by ``--resume`` / saved runs) must
reproduce the configuration exactly, or a resumed run silently differs from the original.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from hypothesis import given, settings
from hypothesis import strategies as st

from src.config.experiment import ExperimentConfig
from src.core.constants import MODEL_FAMILIES

BASE_MODELS = [
    m
    for family in ("boosting", "classical", "neural", "transformer")
    for m in MODEL_FAMILIES[family]
]
META_LEARNERS = MODEL_FAMILIES["meta_learner"]
TRAINING_MODES = ["standard", "walk_forward", "regime_aware", "meta_labeling"]
GAP = st.one_of(st.none(), st.integers(0, 5000))
SAFE_TEXT = st.text(alphabet=st.characters(min_codepoint=32, max_codepoint=126), max_size=20)


@st.composite
def experiment_configs(draw: st.DrawFn) -> ExperimentConfig:
    """A valid ExperimentConfig with drawn models, meta-learner, mode, horizons and gaps."""
    config = ExperimentConfig(
        name=draw(st.text(alphabet="abcdefghij_0123456789", min_size=1, max_size=15)),
        description=draw(SAFE_TEXT),
        run_id=draw(st.text(alphabet="abcdef0123456789_", min_size=1, max_size=20)),
        random_seed=draw(st.integers(0, 2**31 - 1)),
        verbose=draw(st.integers(0, 2)),
    )
    config.data.symbol = draw(st.sampled_from(["MES", "MGC", "MNQ", "ES"]))
    config.data.bar_timeframe = draw(st.sampled_from([None, "1min", "5min", "15min"]))
    config.data.labeling.binary_mode = draw(st.booleans())
    config.data.labeling.upper_mult = draw(st.one_of(st.none(), st.floats(0.5, 5.0)))
    config.data.labeling.max_holding_bars = draw(st.one_of(st.none(), st.integers(1, 200)))
    config.training.models = draw(
        st.lists(st.sampled_from(BASE_MODELS), min_size=1, max_size=6, unique=True)
    )
    config.training.horizons = draw(
        st.lists(st.integers(1, 60), min_size=1, max_size=4, unique=True)
    )
    config.training.meta_learner = draw(st.sampled_from(META_LEARNERS))
    config.training.training_mode = draw(st.sampled_from(TRAINING_MODES))
    config.training.n_splits = draw(st.integers(2, 10))
    config.training.purge_bars = draw(GAP)
    config.training.embargo_bars = draw(GAP)
    config.training.sample_weighting = draw(st.sampled_from(["uniqueness", "none"]))
    config.training.build_ensemble = draw(st.booleans())
    config.evaluation.run_backtest = draw(st.booleans())
    config.evaluation.slippage_ticks = draw(st.one_of(st.none(), st.floats(0.0, 5.0)))
    return config


@settings(max_examples=100, deadline=None)
@given(config=experiment_configs())
def test_dict_roundtrip_is_lossless(config: ExperimentConfig) -> None:
    """to_dict -> from_dict -> to_dict reproduces the original dict exactly."""
    original = config.to_dict()

    restored = ExperimentConfig.from_dict(original)

    assert restored.to_dict() == original
    assert restored.models == config.models
    assert restored.horizons == config.horizons
    assert restored.training.meta_learner == config.training.meta_learner
    assert restored.training.training_mode == config.training.training_mode
    assert restored.output_dir == config.output_dir  # run_id is not appended twice


@settings(max_examples=40, deadline=None)
@given(config=experiment_configs())
def test_yaml_roundtrip_is_lossless(config: ExperimentConfig, tmp_path_factory) -> None:
    """save_yaml -> from_yaml (safe_load) reproduces the configuration."""
    path = Path(tmp_path_factory.mktemp("cfg")) / "config.yaml"

    config.save_yaml(path)
    restored = ExperimentConfig.from_yaml(path)

    assert restored.to_dict() == config.to_dict()
    assert yaml.safe_load(path.read_text()) == config.to_dict()


@settings(max_examples=100, deadline=None)
@given(config=experiment_configs(), bar_timeframe=st.sampled_from(["1min", "5min", "15min"]))
def test_resolved_purge_covers_label_span_and_survives_roundtrip(
    config: ExperimentConfig, bar_timeframe: str
) -> None:
    """Derived CV gaps are non-negative, purge >= longest label span, and stable across a round trip."""
    purge, embargo = config.resolve_cv_gaps(bar_timeframe)
    restored_purge, restored_embargo = ExperimentConfig.from_dict(config.to_dict()).resolve_cv_gaps(
        bar_timeframe
    )

    assert purge >= config.label_span_bars() >= 1
    assert embargo >= 0
    assert (purge, embargo) == (restored_purge, restored_embargo)
