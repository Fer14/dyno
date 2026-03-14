.ONESHELL:
.PHONY: install hooks hooks-update ruff test

SHELL=/bin/bash
RUN=uv run

# Install uv, pre-commit hooks and dependencies
# Note that `$(RUN)` has an implicit `uv sync`, since it will (if necessary):
# - Download an install Python
# - Create a virtual environment
# - Update `uv.lock`
# - Sync the virtual env, installing and removing dependencies as required
install:
	@command -v uv && echo "uv already installed" || curl -LsSf https://astral.sh/uv/install.sh | sh
	$(RUN) pre-commit install

hooks:
	$(RUN) pre-commit run --all-files

hooks-update:
	$(RUN) pre-commit autoupdate

# Specific packages can be upgrade with `--upgrade-package PKG`
upgrade:
	uv lock --upgrade

ruff:
	$(RUN) ruff format .
	$(RUN) ruff check --fix --show-fixes .

test:
	$(RUN) pytest
