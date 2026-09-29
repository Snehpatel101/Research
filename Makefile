.PHONY: help install install-dev test test-quick test-slow lint format type-check dead-code check docs docs-gen docs-check docs-serve examples matrix pre-commit clean

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Create .venv (Python 3.11, CPU torch) and install the package
	uv venv .venv --python 3.11
	uv pip install torch --index-url https://download.pytorch.org/whl/cpu
	uv pip install -e .

install-dev: install ## Install with development dependencies + git hooks
	uv pip install -e ".[dev,stats,docs]"
	pre-commit install

test: ## Run the full test suite (incl. slow end-to-end tests)
	pytest -n auto --dist loadfile

test-quick: ## Run fast tests only (excludes tests marked slow)
	pytest -m "not slow" -n auto --dist loadfile

test-slow: ## Run slow end-to-end tests only
	pytest -m slow -n 2 --dist loadfile

lint: ## Run ruff linter on src/, tests/ and scripts/ (with auto-fix)
	ruff check src/ tests/ scripts/ --fix

format: ## Format code with black
	black src/ tests/ scripts/

type-check: ## Run pyright (must report 0 errors)
	pyright

dead-code: ## Report unused code (vulture, config in pyproject)
	vulture

check: ## Everything CI runs: lint, format, types, dead code, fast tests
	ruff check src/ tests/ scripts/
	black --check src/ tests/ scripts/
	pyright
	vulture
	pytest -m "not slow" -n auto --dist loadfile

docs-gen: ## Regenerate the generated doc pages (configuration, CLI) from the code
	python scripts/gen_config_docs.py
	python scripts/gen_cli_docs.py

docs-check: ## Fail if the generated doc pages are stale
	python scripts/gen_config_docs.py --check
	python scripts/gen_cli_docs.py --check

docs: docs-check ## Build the documentation site (strict: any warning fails)
	mkdocs build --strict

docs-serve: ## Serve the documentation site with live reload on http://127.0.0.1:8000
	mkdocs serve

examples: ## Run the three example scripts (a few minutes each on a CPU)
	for f in examples/0*.py; do OMP_NUM_THREADS=1 python $$f || exit 1; done

matrix: ## Mix-and-match matrix: every model solo and in pairs, every meta-learner and mode
	for k in solo pairs meta modes modes-solo binary all-in; do python scripts/mix_match.py $$k --jobs 3; done
	python scripts/mix_match.py report

pre-commit: ## Run all pre-commit hooks on all files
	pre-commit run --all-files

pre-commit-update: ## Update pre-commit hook versions
	pre-commit autoupdate

clean: ## Remove build artifacts and cache files
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf build/ dist/ .coverage htmlcov/
