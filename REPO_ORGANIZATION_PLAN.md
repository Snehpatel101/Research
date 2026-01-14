# Repository Organization & Unification Plan

This document outlines a staged approach to organizing the `Research` repository to improve discoverability, reduce clutter, and unify documentation standards.

## 🎯 Objectives
1.  **Declutter Root:** Keep the root directory clean, containing only essential configuration and entry points.
2.  **Unify Documentation:** Centralize knowledge in `docs/` with a clear hierarchy.
3.  **Clarify Configuration:** Explicitly document the "Source of Truth" for configuration to avoid confusion.
4.  **Standardize Planning:** Consolidate scattered roadmaps and findings into a central planning directory.

---

## 📅 Phase 1: Root Directory Cleanup
**Goal:** Remove noise from the primary workspace.

| Action | Source | Destination | Notes |
|:-------|:-------|:------------|:------|
| **Move** | `FINDINGS 1.MD` | `docs/analysis/findings_part1.md` | Standardize naming (lowercase). |
| **Move** | `FINDINGS 2.md` | `docs/analysis/findings_part2.md` | Preserves recent context in analysis. |
| **Move** | `JUPYTER NOTEBOOK BASICS(READ).md` | `docs/guides/jupyter_basics.md` | Rename for clarity and consistency. |
| **Migrate** | `SNEH IMPROVEMENT PLAN/` | `docs/planning/legacy_roadmap/` | Move entire folder to planning archive or consolidate. |
| **Verify** | `CLAUDE.md` | *Keep in Root* (Context) | Keep if used by AI agents, otherwise move to `.serena/`. |
| **Clean** | `THINGS TO HANDLE AFTER THE REPO IS ORGANIZED./` | `docs/planning/backlog/` | Move contents to a structured backlog. |

**Result:** Root directory should only contain:
- `src/`, `data/`, `docs/`, `tests/`, `config/`, `notebooks/`
- `requirements*.txt`, `setup.py`, `pyproject.toml`
- `.gitignore`, `README.md`, `CLAUDE.md`, `MANIFEST.in`, `pytest.ini`
- `pipeline` (executable)
- Hidden dirs: `.git`, `.dvc`, `.github`, `.serena`, `.venv`, `.vscode`

---

## 📂 Phase 2: Documentation Restructuring
**Goal:** Make documentation easy to navigate and maintain.

1.  **Consolidate Top-Level Docs:**
    - Move `docs/*.md` files into subdirectories where possible.
    - Example: `docs/MLOPS_*.md` -> `docs/guides/mlops/`
    - Example: `docs/TRADING_SIMULATOR_QUICKSTART.md` -> `docs/guides/trading/`

2.  **Update Index:**
    - Refresh `docs/README.md` to link to the new locations.
    - Ensure `docs/INDEX.md` (if exists) matches the folder structure.

3.  **Standardize Naming:**
    - Enforce `snake_case` or `kebab-case` for filenames.
    - Remove caps-lock filenames (e.g., `FEATURE_SELECTION_BY_ARCHITECTURE.md` -> `feature_selection_architecture.md`).

---

## ⚙️ Phase 3: Configuration Clarity
**Goal:** Resolve the "Multiple Config Sources" ambiguity (as identified in Findings 2).

1.  **Create Truth Document:**
    - Create `docs/reference/CONFIGURATION_TRUTH.md`.
    - **Content:** Explicitly state that `src/phase1/config/` and `pipeline_config.py` are the **primary** sources for the pipeline.
    - Mark `config/*.yaml` as "Reference/Legacy" or "Template" if they are not actively loaded by the pipeline code.

2.  **Deprecation Notice (Documentation Only):**
    - Add a `README.md` in `config/` explaining which files are active and which are decorative/examples.

---

## 🚀 Phase 4: Workflow & Planning Unification
**Goal:** Centralize project management.

1.  **Unified Roadmap:**
    - Create `docs/planning/MASTER_ROADMAP.md`.
    - Merge items from `SNEH IMPROVEMENT PLAN` and `THINGS TO HANDLE...` into this master roadmap.
    - Categorize into "Now", "Next", "Later".

2.  **Knowledge Base Integration:**
    - Review `.serena/knowledge/` and `docs/` for overlaps.
    - Ensure architectural decisions are in `docs/architecture/` (or `docs/reference/`).

---

## ✅ Execution Checklist

- [ ] **Phase 1:** Move root files (`FINDINGS`, `JUPYTER...`, `SNEH...`, `THINGS TO HANDLE...`) to `docs/`.
- [ ] **Phase 2:** Reorganize `docs/` folder into semantic subfolders (`guides`, `reference`, `planning`).
- [ ] **Phase 3:** Write `docs/reference/CONFIGURATION_TRUTH.md` detailing the config hierarchy.
- [ ] **Phase 4:** Consolidate scattered tasks into `docs/planning/MASTER_ROADMAP.md`.
- [ ] **Final:** Update root `README.md` to reflect the new structure.
