# CPU-only inference image. Build:  docker build -t screenpad .
# Run:    docker run --rm -v /path/to/images:/data/in -v /path/to/out:/data/out screenpad \
#             infer --input /data/in --output /data/out/scores.csv
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.8.10 /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/opt/venv PATH=/opt/venv/bin:$PATH

WORKDIR /app
# Dependencies first so code edits don't invalidate the (large) torch layer.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --extra torch --no-install-project

COPY README.md ./
COPY screenpad ./screenpad
COPY configs ./configs
COPY artefacts ./artefacts
RUN uv sync --frozen --no-dev --extra torch

ENTRYPOINT ["screenpad"]
CMD ["--help"]
