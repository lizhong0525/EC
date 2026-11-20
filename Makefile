.PHONY: install sync lock test coverage-core lint format typecheck schemas secret-scan check web-install web-test web-build

PYTHON_BOOTSTRAP ?= python3.11
PYTHON := .venv/bin/python
PYTEST := PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/pytest -p pytest_asyncio.plugin
UV ?= .venv/bin/uv

install:
	@test ! -e .venv || { echo ".venv already exists; remove it explicitly before a clean install" >&2; exit 1; }
	$(PYTHON_BOOTSTRAP) -m venv .venv
	$(PYTHON) -m pip install --require-hashes -r requirements-dev.lock
	$(PYTHON) -m pip install --no-deps -e .

sync:
	@test -x $(PYTHON) || { echo ".venv is missing; run 'make install' with Python 3.11" >&2; exit 1; }
	@$(PYTHON) -c "import sys; assert sys.version_info[:2] == (3, 11), 'EcoCompanion requires a Python 3.11 virtual environment'"
	$(PYTHON) -m pip install --require-hashes -r requirements-dev.lock
	$(PYTHON) -m pip install --no-deps -e .

lock:
	$(UV) pip compile pyproject.toml \
		--python-version 3.11 \
		--python-platform x86_64-manylinux_2_17 \
		--exclude-newer 2026-09-03T23:59:59Z \
		--generate-hashes \
		--no-emit-package ecocompanion \
		--custom-compile-command "make lock" \
		--output-file requirements.lock
	$(UV) pip compile pyproject.toml \
		--extra dev \
		--python-version 3.11 \
		--python-platform x86_64-manylinux_2_17 \
		--exclude-newer 2026-09-03T23:59:59Z \
		--generate-hashes \
		--no-emit-package ecocompanion \
		--custom-compile-command "make lock" \
		--output-file requirements-dev.lock

test:
	$(PYTEST) -q

coverage-core:
	$(PYTEST) \
		-p pytest_cov.plugin \
		--cov=ecocompanion.core \
		--cov=ecocompanion.domain \
		--cov=ecocompanion.agent.safety \
		--cov-report=term-missing \
		--cov-fail-under=85 \
		-q

lint:
	.venv/bin/ruff check backend/src backend/tests

format:
	.venv/bin/ruff format backend/src backend/tests
	.venv/bin/ruff check --fix backend/src backend/tests

typecheck:
	.venv/bin/mypy backend/src/ecocompanion

schemas:
	$(PYTHON) -c "from pathlib import Path; from ecocompanion.schema_export import export_schemas; export_schemas(Path('artifacts/schemas'))"

secret-scan:
	$(PYTHON) scripts/secret_scan.py .

web-install:
	npm --prefix web install

web-test:
	npm --prefix web test

web-build:
	npm --prefix web run build

check: lint typecheck test coverage-core schemas secret-scan web-test web-build
