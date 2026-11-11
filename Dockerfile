# syntax=docker/dockerfile:1
# lite-mem's HTTP service (design §7). Models are not in the image: Ollama (or an OpenAI-compatible
# endpoint) is reached over the network, by default on the Docker host.
FROM python:3.12-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.10.8 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
# Dependencies first, so source edits do not invalidate this layer.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked --no-dev --extra server --no-install-project
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked --no-dev --extra server --no-editable

FROM python:3.12-slim
RUN useradd --system --create-home --uid 10001 litemem
COPY --from=build /app/.venv /app/.venv
ENV PATH=/app/.venv/bin:$PATH PYTHONUNBUFFERED=1
USER litemem
WORKDIR /home/litemem
EXPOSE 8710
# /health checks the database only: a frequent probe must not keep the model loaded.
# `/health?deep=true` also embeds once (what a caller asks before relying on the service).
HEALTHCHECK --interval=10s --timeout=5s --start-period=10s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8710/health', timeout=4)"]
CMD ["sh", "-c", "litemem migrate && exec litemem serve --host 0.0.0.0"]
