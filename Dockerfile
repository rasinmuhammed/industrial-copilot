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
    COPILOT_DB=/tmp/copilot.duckdb

USER copilot
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

CMD ["uvicorn", "copilot.api:app", "--host", "0.0.0.0", "--port", "8000"]
