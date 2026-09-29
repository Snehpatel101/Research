"""
Generate docs/configuration.md from the ExperimentConfig dataclasses.

Every field, type and default is read from the code, so the page cannot drift
from what ``ExperimentConfig`` actually accepts. Descriptions come from, in
order: the class docstring's ``Attributes:`` section, a trailing ``# comment``
on the field line, the ``FALLBACK`` table below, then a comment block directly
above the field. Allowed values are read from the enums / constants the
pipeline validates against.

Usage:
    python scripts/gen_config_docs.py            # write docs/configuration.md
    python scripts/gen_config_docs.py --check    # exit 1 if the page is stale
"""

from __future__ import annotations

import argparse
import dataclasses
import inspect
import re
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.config.experiment import ExperimentConfig  # noqa: E402

OUT = REPO_ROOT / "docs" / "configuration.md"

# Descriptions for fields whose source carries none (or only a section header
# comment). Keys are dotted paths from the ExperimentConfig root.
FALLBACK: dict[str, str] = {
    "data.symbol": (
        "Contract symbol. Selects the barrier table, tick size/value, costs, session "
        "hours and ADX regime threshold (MES, MGC and MNQ have presets; other symbols "
        "use defaults)."
    ),
    "data.data_path": "Raw OHLCV bars (parquet or csv) with a datetime index or column.",
    "data.start_date": "Keep bars at or after this timestamp (UTC).",
    "data.end_date": "Keep bars at or before this timestamp (UTC).",
    "data.features": "Feature selection switch.",
    "data.labeling": "Triple-barrier labeling.",
    "data.sequence": "Sequence window for 3D/4D models.",
    "data.mtf": "Multi-timeframe features and 4D streams.",
    "data.splits": "Chronological train/val/test ratios.",
    "training.models": "Base models to train (any subset of the registered models).",
    "training.horizons": (
        "Label horizons. Each horizon keys the barrier table (k_up, k_down, max_bars); "
        "the first one drives the backtest and the default `label` column."
    ),
    "training.cv_method": (
        "CV for hyperparameter tuning. OOF stacking always uses purged k-fold; "
        "`cpcv` makes the Optuna tuner score trials on CPCV paths."
    ),
    "training.training_mode": (
        "How models are trained and evaluated (see Concepts: walk-forward, regime, "
        "meta-labeling)."
    ),
    "training.purge_bars": (
        "Bars dropped before every test block. None = the longest label span "
        "(max_bars over the horizons); an explicit value shorter than that is raised "
        "to it. CV additionally purges on every label's actual end bar."
    ),
    "training.regime.detection_method": "Rolling statistic that assigns each bar a regime.",
    "training.meta_labeling.meta_model": (
        "Classifier that learns P(the primary's bet pays off). The primary model is "
        "`training.models[0]`."
    ),
    "training.meta_labeling.threshold": (
        "Take a primary bet only when the meta-model's P(bet pays off) is at least this."
    ),
    "training.n_splits": "Folds for purged k-fold (OOF predictions, tuning).",
    "training.embargo_bars": (
        "Bars dropped after every test block. None = one trading day of bars at the "
        "bar timeframe, capped at 25% of a CV fold."
    ),
    "training.walk_forward": "Walk-forward windows (training_mode=walk_forward).",
    "training.regime": "Regime-aware mode (training_mode=regime_aware).",
    "training.meta_labeling": "Meta-labeling mode (training_mode=meta_labeling).",
    "training.optuna": "Hyperparameter tuning budget.",
    "training.calibration": "Probability calibration.",
    "training.batch_size": "Neural batch size (device is auto-detected per model).",
    "training.max_epochs": "Neural epoch budget (early stopping usually ends sooner).",
    "training.early_stopping_patience": "Epochs without validation improvement before stopping.",
    "training.build_ensemble": "Stack the base models' OOF predictions with `meta_learner`.",
    "training.meta_learner": "Stacking meta-learner.",
    "evaluation.run_backtest": "Backtest the deployed strategy's out-of-sample signals.",
    "evaluation.position_sizing": (
        "Backtest sizing: `fixed` (fixed contracts), `volatility` (volatility targeted), "
        "`confidence` (bet sizing from model confidence), `kelly`."
    ),
    "evaluation.commission_per_contract": (
        "Override the round-trip commission per contract (dollars; exchange and NFA "
        "fees are added on top)."
    ),
    "evaluation.slippage_ticks": "Override the slippage per fill (ticks, one way).",
    "evaluation.initial_equity": "Starting equity of the backtest (dollars).",
    "bundling.create_bundle": "Write one inference bundle per model (and ensemble) per horizon.",
    "bundling.deploy_artifact": "Write `deploy/manifest.json` indexing the bundles.",
}

# Header comments that label a group of fields rather than describe one.
_HEADER_COMMENT = re.compile(r"^[A-Z][\w /-]*$")
# Inline comments that only list allowed values ("standard, walk_forward, ...")
_ENUMERATION = re.compile(r"^[\w]+(, [\w]+)+$")


def _choices() -> dict[str, list[str]]:
    """Allowed values, read from what the pipeline validates against."""
    import src.models  # noqa: F401 - registers every model
    from src.config.cv import WindowType
    from src.config.training import CALIBRATION_METHODS, OPTUNA_METRICS
    from src.core.config import SAMPLE_WEIGHTING_MODES
    from src.core.types import CVMethod, TrainingMode
    from src.models.registry import ModelRegistry
    from src.models.training.regime_detector import RegimeDetectionMethod

    families = ModelRegistry.list_models()
    base = sorted(
        m
        for fam, names in families.items()
        if fam not in ("ensemble", "meta_learner")
        for m in names
    )
    return {
        "training.models": base,
        "training.training_mode": [m.value for m in TrainingMode],
        "training.cv_method": [m.value for m in CVMethod],
        "training.sample_weighting": list(SAMPLE_WEIGHTING_MODES),
        "training.meta_learner": sorted(families.get("meta_learner", [])),
        "training.walk_forward.window_type": [w.value for w in WindowType],
        "training.regime.detection_method": [m.value for m in RegimeDetectionMethod],
        "training.regime.n_regimes": ["2", "3"],
        "training.meta_labeling.meta_model": [
            "logistic",
            "random_forest",
            "xgboost",
            "lightgbm",
            "catboost",
        ],
        "training.optuna.metric": list(OPTUNA_METRICS),
        "training.calibration.method": list(CALIBRATION_METHODS),
        "evaluation.position_sizing": ["fixed", "volatility", "confidence", "kelly"],
    }


def _docstring_attributes(cls: type) -> dict[str, str]:
    """``name: description`` entries of the docstring's Attributes section."""
    doc = inspect.getdoc(cls) or ""
    out: dict[str, str] = {}
    in_attrs = False
    current: str | None = None
    for line in doc.splitlines():
        if line.strip() == "Attributes:":
            in_attrs = True
            continue
        if not in_attrs:
            continue
        if line and not line.startswith(" "):
            break  # next section (Example:, etc.)
        m = re.match(r"^ {4}(\w+): (.*)$", line)
        if m:
            current = m.group(1)
            out[current] = m.group(2).strip()
        elif current and line.startswith("        ") and line.strip():
            out[current] += " " + line.strip()
        elif not line.strip():
            current = None
    return out


def _source_comments(cls: type) -> tuple[dict[str, str], dict[str, str]]:
    """(trailing comment, comment block directly above) per field of ``cls``."""
    inline: dict[str, str] = {}
    above: dict[str, str] = {}
    block: list[str] = []
    for raw in inspect.getsource(cls).splitlines():
        line = raw.strip()
        if line.startswith("#"):
            block.append(line.lstrip("#").strip())
            continue
        m = re.match(r"^(\w+): [^=]+(?:= (.*))?$", line)
        if m and raw.startswith("    ") and not raw.startswith("        "):
            name = m.group(1)
            tail = raw.split("#", 1)
            if len(tail) == 2 and "(" not in tail[1][:1]:
                inline[name] = tail[1].strip()
            if block and not (len(block) == 1 and _HEADER_COMMENT.match(block[0])):
                above[name] = " ".join(block)
        block = []
    return inline, above


def _type_name(f: dataclasses.Field) -> str:
    return f.type if isinstance(f.type, str) else getattr(f.type, "__name__", str(f.type))


def _default(f: dataclasses.Field, path: str) -> str:
    if path == "run_id":
        return "timestamp + random suffix"
    if f.default is not dataclasses.MISSING:
        value = f.default
    elif f.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
        value = f.default_factory()  # type: ignore[misc]
    else:
        return "required"
    if dataclasses.is_dataclass(value):
        return "see below"
    if path == "output_dir":
        return "`experiments/runs` (+ `/<run_id>`)"
    return f"`{value!r}`" if not isinstance(value, str) else f'`"{value}"`'


def _escape(text: str) -> str:
    return text.replace("|", "\\|")


def _section(
    cls: type,
    path: str,
    choices: dict[str, list[str]],
    missing: list[str],
    out: list[str],
) -> None:
    attrs = _docstring_attributes(cls)
    inline, above = _source_comments(cls)
    nested: list[tuple[type, str]] = []
    title = path or "ExperimentConfig (top level)"
    summary = (inspect.getdoc(cls) or "").split("\n\n")[0].replace("\n", " ")
    out += [f"## `{title}`", "", f"{summary} (`{cls.__module__}.{cls.__name__}`)", ""]
    out += ["| Field | Type | Default | Description |", "|---|---|---|---|"]
    for f in dataclasses.fields(cls):
        dotted = f"{path}.{f.name}" if path else f.name
        factory = f.default_factory
        is_section = isinstance(factory, type) and dataclasses.is_dataclass(factory)
        comment = inline.get(f.name, "")
        if dotted in choices and _ENUMERATION.match(comment):
            comment = ""  # the Choices list below says the same, from the code
        desc = attrs.get(f.name) or comment or FALLBACK.get(dotted) or above.get(f.name) or ""
        if not desc:
            missing.append(dotted)
        desc = desc.strip()
        if desc and desc[-1] not in ".)":
            desc += "."
        if dotted in choices:
            desc += " Choices: " + ", ".join(f"`{c}`" for c in choices[dotted]) + "."
        if is_section:
            anchor = dotted.replace(".", "")
            desc = f"{desc} See [`{dotted}`](#{anchor})."
            nested.append((factory, dotted))  # type: ignore[arg-type]
        out.append(
            f"| `{f.name}` | `{_escape(_type_name(f))}` | {_default(f, dotted)} "
            f"| {_escape(desc.strip())} |"
        )
    out.append("")
    for sub_cls, sub_path in nested:
        _section(sub_cls, sub_path, choices, missing, out)


def render() -> tuple[str, list[str]]:
    """Markdown for the configuration page, plus fields without a description."""
    out = [
        "# Configuration reference",
        "",
        "<!-- Generated by scripts/gen_config_docs.py from the ExperimentConfig dataclasses."
        " Do not edit by hand: run `make docs-gen`. -->",
        "",
        "One object configures a run: `ExperimentConfig`. Set fields in Python, or load a YAML",
        "file with `ExperimentConfig.from_yaml(path)`; every run saves the config it used as",
        "`<run_dir>/experiment_config.yaml`, which loads back to an identical config.",
        "Unknown keys (typos, fields removed in later versions) are logged and ignored.",
        "",
        "```python",
        "from src.config.experiment import ExperimentConfig",
        "",
        'cfg = ExperimentConfig(name="mes_h5")',
        'cfg.data.data_path = "data/raw/MES_1m.parquet"',
        'cfg.training.models = ["xgboost", "lstm"]',
        "cfg.training.optuna.n_trials = 0",
        "```",
        "",
        "A `None` default on the barrier multipliers, `max_holding_bars`, `purge_bars`,",
        "`embargo_bars` and `seq_len` means *derived*: barriers come from the per-symbol",
        "barrier table, purge and embargo from the label span and the bar timeframe, the",
        "sequence length from each model's contract (see [Concepts](concepts.md)).",
        "",
    ]
    missing: list[str] = []
    _section(ExperimentConfig, "", _choices(), missing, out)

    defaults = ExperimentConfig().to_dict()
    for key in ("run_id", "output_dir"):
        defaults.pop(key, None)
    out += [
        "## Full default config as YAML",
        "",
        "`ExperimentConfig().save_yaml(path)` writes this (plus `run_id` and `output_dir`):",
        "",
        "```yaml",
        yaml.safe_dump(defaults, default_flow_style=False, sort_keys=False).rstrip(),
        "```",
        "",
    ]
    return "\n".join(out), missing


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--check", action="store_true", help="fail if the page is stale")
    args = parser.parse_args()

    import logging

    logging.disable(logging.WARNING)
    text, missing = render()
    if missing:
        print(f"warning: fields without a description: {missing}", file=sys.stderr)
    if args.check:
        current = OUT.read_text() if OUT.exists() else ""
        if current != text:
            sys.exit(f"{OUT.relative_to(REPO_ROOT)} is stale: run `make docs-gen`")
        print(f"{OUT.relative_to(REPO_ROOT)} is up to date")
        return
    OUT.write_text(text)
    print(f"Wrote {OUT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
