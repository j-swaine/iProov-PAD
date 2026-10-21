# CPU-only inference image. Model weights are not baked in: on first use they are downloaded (~2 GB) from the public
# Hugging Face repo into the /models volume and reused by every later run.
#
# Build:      docker build -t screenpad .
# Check:      docker run --rm -v screenpad-models:/models screenpad preflight
# Score:      docker run --rm -v screenpad-models:/models -v /path/to/images:/data/in:ro -v /path/to/out:/data/out \
#                 screenpad infer --input /data/in --output /data/out/scores.csv
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.8.10 /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/opt/venv PATH=/opt/venv/bin:$PATH

WORKDIR /app
# Dependencies first so code edits don't invalidate the (large) torch layer.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --extra torch --no-install-project

COPY README.md infer.py ./
COPY screenpad ./screenpad
COPY configs ./configs
COPY artefacts ./artefacts
RUN uv sync --frozen --no-dev --extra torch

# `infer` looks for weights in ./models (its default): point that at the volume
ENV HF_HOME=/models/hf
RUN ln -s /models /app/models
VOLUME ["/models"]
ENTRYPOINT ["screenpad"]
CMD ["--help"]
