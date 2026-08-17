.DEFAULT_GOAL := help
PY := .venv/bin/python
PIP := .venv/bin/pip

.PHONY: help venv install build verify discover bench clean

help:  ## Show available targets
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "  See docs/13-BUILD-PLAN.md."

venv:  ## Create the virtualenv
	python3 -m venv .venv

install: venv  ## Install dependencies
	$(PIP) install -q --upgrade pip
	$(PIP) install -e .

build:  ## CSV -> DuckDB with physics columns and margins
	$(PY) -m copilot.ingest

verify:  ## Reproduce every documented figure, incl. the README example  (CI gate)
	$(PY) scripts/verify_dataset.py
	$(PY) scripts/verify_readme.py

discover:  ## Re-derive the documented thresholds from data alone
	$(PY) scripts/discover_rules.py

exemplars:  ## Seed data/exemplars.jsonl for the distillation notebook
	$(PY) scripts/export_exemplars.py

sft:  ## Export verified (question, plan) pairs for SLM fine-tuning
	$(PY) scripts/export_sft_data.py > data/sft_train.jsonl

notebook:  ## Dry-run the Kaggle training notebook on the CPU path
	$(PY) scripts/check_notebook.py

calibrate:  ## Measure the exemplar-retrieval thresholds
	$(PY) scripts/calibrate_exemplars.py

bench:  ## Margin evaluation throughput
	$(PY) scripts/bench.py

bench-slm:  ## Benchmark the SLM planner against the grammar tier
	$(PY) scripts/bench_slm.py

chat:  ## Interactive terminal copilot
	$(PY) -m copilot.cli chat

ask:  ## One question: make ask Q="why did cycle 9016 fail?"
	$(PY) -m copilot.cli ask "$(Q)"

demo:  ## Scripted walkthrough of all four acceptance criteria
	$(PY) -m copilot.cli demo

serve:  ## API, console and envelope explorer on :8000
# A held-open SSE stream blocks uvicorn's graceful shutdown indefinitely —
# the reloader sits at "Waiting for connections to close" until the replay
# finishes, which is 13 minutes with one browser tab open. The same wait
# stalls a redeploy in production, so shutdown is bounded, not patient.
	.venv/bin/uvicorn copilot.api:app --reload --port 8000 --timeout-graceful-shutdown 3

stream:  ## Replay the fleet as an event stream with live alerts
	$(PY) -c "from copilot.stream import replay, StreamScorer; s=StreamScorer(); \
[print(f'{a.machine_id:<6} {a.mode:<4} {a.kind.value:<12} ' + (f'crosses in ~{a.lead_time_min:.0f} min' if a.lead_time_min else a.message[:60])) \
 for t in replay(scorer=s, speed=0) for a in t.alerts]; \
print(f'\\n{s.ticks:,} cycles, {s.alerts_raised} alerts, {s.alerts_suppressed} suppressed')"

eval:  ## Run the golden eval suite (hard gates fail the build)
	$(PY) evals/run_evals.py

eval-json:  ## Machine-readable eval report
	$(PY) evals/run_evals.py --json

docker:  ## Build the deployment image
	docker build -t industrial-copilot .

serve-docker:  ## Run the copilot in a container on :8000
	docker compose up --build

outcomes:  ## Score the streaming alerts against what actually happened
	$(PY) scripts/score_outcomes.py

coverage:  ## Risk-outcomes: how much do we answer, and how sound is it
	$(PY) evals/coverage.py

onboard:  ## Discover a process definition from a dataset, and audit it
	$(PY) scripts/onboard.py --csv data/ai4i2020.csv --label "Machine failure" \
		--audit --out artifacts/discovered.yaml

metamorphic:  ## Relations that must hold, checked on generated inputs
	$(PY) -m pytest tests/test_metamorphic.py -q

adversarial:  ## Run only the adversarial suite
	$(PY) -m pytest tests/test_adversarial.py -q

test:  ## Run the test suite
	$(PY) -m pytest tests/ -q

clean:  ## Remove generated artifacts
	rm -f data/warehouse.duckdb
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
