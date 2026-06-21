PYTHON ?= python
CONFIG ?= config/default.yaml

.PHONY: install sample generate validate test clean

install:
	$(PYTHON) -m pip install -e ".[test]"

sample:
	$(PYTHON) scripts/generate_sample.py

generate:
	$(PYTHON) -m simulation_data_generation.cli generate --config $(CONFIG)

validate:
	$(PYTHON) -m simulation_data_generation.cli validate --data-dir data/generated

test:
	$(PYTHON) -m pytest

clean:
	$(PYTHON) -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in [pathlib.Path('data/generated'), pathlib.Path('reports/figures')]]"
	$(PYTHON) -c "import pathlib; pathlib.Path('data/generated').mkdir(parents=True, exist_ok=True); pathlib.Path('reports/figures').mkdir(parents=True, exist_ok=True)"

