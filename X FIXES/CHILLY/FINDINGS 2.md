# HISTORICAL - Configuration Findings

> **SUPERSEDED:** This document is historical. See `CLAUDE.md` and `docs/planning/PROJECT_CHARTER.md` for current status.

---

I took a deeper pass focused on configuration and structure. Here's a clearer map of what's real vs. what looks unused or conflicting, without changing anything.

Config Map (Source vs. Surface)

Phase‑1 pipeline config is the real source of truth: pipeline_config.py plus validation/defaults in src/phase1/config/*. This is what the pipeline stages consume.
Model training is driven by TrainerConfig in trainer_config.py, but most entrypoints construct it directly (no YAML merge).
YAMLs under *.yaml are only used if you call the config merger in merging.py; I don’t see that wired into CLI or training utils.
Global YAMLs training.yaml and cv.yaml are loadable via loaders.py, but I don’t see any code path that actually calls load_training_config() or load_cv_config().
Disorganization / Mismatch Hotspots

Multiple config systems coexist but don’t connect: config/ YAMLs vs. PipelineConfig dataclass. That’s likely why you feel “disorganized.”
training.yaml defaults to two symbols (MES, MGC), but the pipeline enforces single‑symbol runs in pipeline_config.py.
cv.yaml uses embargo: 60, but the rest of the code assumes 1440 bars for 5‑min data. If cv.yaml ever gets wired in, that’s leakage risk.
Feature/ensemble/experiment YAMLs (config/features/*, config/ensembles/*, config/experiments/*) don’t appear to be referenced anywhere. They’re effectively documentation unless you wire them in.
The “config file in src” you’re thinking of is probably pipeline_config.py. That’s the only config that’s definitely used.
What I’d recommend (no code changes, just clarity)

Declare one source of truth for each phase: Pipeline → pipeline_config.py, Training → TrainerConfig or YAMLs, but not both without explicit wiring.
If you want to keep YAMLs, write down a short “config entrypoint” doc so you remember which file actually changes behavior (even a checklist in a note file).
Decide whether config/* are templates or live configs; if they’re templates, treat them as docs.