.DEFAULT_GOAL := help
PY := .venv/bin/python
PIP := .venv/bin/pip

.PHONY: help venv install build verify discover bench clean

help:  ## Show available targets
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "  Phase 4+ targets (serve, stream, eval) land as the engine is built."
	@echo "  See docs/13-BUILD-PLAN.md."

venv:  ## Create the virtualenv
	python3 -m venv .venv

install: venv  ## Install dependencies
	$(PIP) install -q --upgrade pip
	$(PIP) install -e .

build:  ## CSV -> DuckDB with physics columns and margins
	$(PY) -m copilot.ingest

verify:  ## Reproduce every figure in docs/01-DATASET.md  (CI gate)
	$(PY) scripts/verify_dataset.py

discover:  ## Re-derive the documented thresholds from data alone
	$(PY) scripts/discover_rules.py

bench:  ## Margin evaluation throughput
	$(PY) scripts/bench.py

chat:  ## Interactive terminal copilot
	$(PY) -m copilot.cli chat

ask:  ## One question: make ask Q="why did cycle 9016 fail?"
	$(PY) -m copilot.cli ask "$(Q)"

demo:  ## Scripted walkthrough of all four acceptance criteria
	$(PY) -m copilot.cli demo

test:  ## Run the test suite
	$(PY) -m pytest tests/ -q

clean:  ## Remove generated artifacts
	rm -f data/warehouse.duckdb
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
