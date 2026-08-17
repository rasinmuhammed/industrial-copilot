# Industrial Copilot — single-node deployment.
#
# Deliberately boring. No CUDA, no model weights baked in, no cloud SDK. The
# whole system answers from the grammar tier with every API key unset, so the
# image needs Python, seven wheels, and the process definition. That property is
# worth more in a plant than anything clever would be: an air-gapped site can
# run this, and nothing about a question leaves the box.
#
#   docker build -t industrial-copilot .
#   docker run -p 8000:8000 -v "$PWD/data:/app/data:ro" industrial-copilot
#
# The LLM planner tier is optional and off unless a key is supplied. Without one
# the engine still answers every golden question — it just declines the long
# tail instead of guessing at it.

FROM python:3.12-slim AS build

WORKDIR /app
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1

# Dependency layer first, so source edits do not re-resolve the wheel set.
COPY pyproject.toml README.md ./
COPY copilot/__init__.py copilot/__init__.py
RUN python -m venv /opt/venv && /opt/venv/bin/pip install --upgrade pip \
 && /opt/venv/bin/pip install .

COPY . .
RUN /opt/venv/bin/pip install --no-deps .

# Build the warehouse at IMAGE BUILD time, not at boot.
#
# The image ships the CSV and not the DuckDB file — the warehouse is derived, so
# shipping it would mean shipping the output instead of the recipe. But nothing
# then built it, so the container started, the first request called
# Engine.build(), and there was no warehouse to open. The platform healthcheck
# hit /health, got no answer, and retired the replica: "1/1 replicas never
# became healthy".
#
# Doing it here rather than in an entrypoint means boot is a process start
# rather than an ingest, so the healthcheck passes in the window platforms
# actually allow. It costs about three seconds of build and 4.5 MB of layer.
RUN /opt/venv/bin/python -m copilot.ingest


FROM python:3.12-slim

# Run as a non-root user. A copilot reads process history; it has no business
# holding root in a plant network.
RUN useradd --create-home --uid 10001 copilot
WORKDIR /app

COPY --from=build /opt/venv /opt/venv
COPY --from=build --chown=copilot:copilot /app /app

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    COPILOT_DB=/app/data/warehouse.duckdb \
    PORT=8000

USER copilot
# Documentation only. The port actually bound is $PORT, below.
EXPOSE 8000

# Liveness is not "the port is open". The engine is only useful once the process
# definition has loaded and the physics evaluates, so the check exercises both.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "\
from copilot.process_model import load_process_model; \
from copilot.physics import OperatingPoint, evaluate; \
m = load_process_model(); \
assert m.deterministic_modes, 'no process definition loaded'; \
evaluate(OperatingPoint(300.0, 310.0, 1500.0, 40.0, 100.0, 'L')); \
print('ok')" || exit 1

# Shell form on purpose: $PORT has to expand.
#
# Every managed platform assigns the port and routes to it. With `--port 8000`
# hardcoded, the app listened on 8000, the platform probed the port it had
# assigned, and every healthcheck attempt returned service unavailable while the
# process sat there running perfectly.
#
# --timeout-graceful-shutdown because an open SSE stream otherwise holds the old
# container through the whole replay and stalls the next deploy.
CMD uvicorn copilot.api:app --host 0.0.0.0 --port ${PORT:-8000} --timeout-graceful-shutdown 5
