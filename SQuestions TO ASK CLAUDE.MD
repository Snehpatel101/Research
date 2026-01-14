# Ask claude to do this FIRST.

1. "Use 3 specialized parallel agents to audit and validate the claims in `SNEH.md` against the actual repository and `src/` pipeline. For each claim, determine whether it is accurate, partially accurate, or incorrect, and cite the specific code locations or docs that support the conclusion.

Then use specialized sequential agents (each passing clear context and decisions forward to the next) to implement fixes for every validated issue from `SNEH.md`. Preserve the heart of the program: do not remove or add core functionality we rely on. The goal is a cleaner, more unified, more consistent codebase with verified end-to-end compatibility. Include tests, integration checks, and a verification summary documenting what changed and how you confirmed it works."



# After claude fixes SNEH.MD.
***QUESTIONS FOR CLAUDE***
1. Use 3–4 specialized parallel agents to deeply review the `src/` pipeline and the overall repository. Identify inconsistencies, architectural mismatches, duplicated responsibilities, unclear boundaries, and any logic that contradicts the intended design. Deliver the findings as a structured root-level document titled `FIXES2.md` (clear sections, ranked severity, concrete examples, and recommended fixes).

2. Use parallel specialized agents to propose the highest-impact cleanup and organizational improvements for the codebase (repo structure, module boundaries, configuration, naming, data flow, and responsibility splits). Append these recommendations to `FIXES2.md`, emphasizing changes that reduce complexity, remove legacy/unused code paths, and improve maintainability without altering core behavior.

3. Use sequential specialized agents to take the issues and recommendations in `FIXES2.md` and implement the fixes end-to-end while preserving the spirit of the program. Ensure each change is validated with tests, integration checks, and sanity runs so the system still works as one cohesive pipeline. Document what changed, why, and how it was verified.

4. Use parallel specialized agents to produce a concise “organizational refactor plan” for the program: what to merge, what to split, what to delete, and how to restructure folders/modules for long-term clarity and extensibility. Write it as a root-level document (or append to `FIXES2.md` if preferred), with an explicit step-by-step execution order and rollback-safe checkpoints.

5. Deeply analyze `src/` and the program’s intended goals (real OHLCV model training, multi-timeframe datasets, feature engineering, ensemble support, and reproducible evaluation). Identify the biggest improvements required to reliably train real single models and ensembles. List missing components (feature sets, labeling, leakage controls, data validation, experiment tracking, model registry, evaluation tooling, etc.) and prioritize them by impact and effort. Use online research too.
