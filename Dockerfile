# Inference image for the GPU Power Modeling API.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install dependencies first for better layer caching.
COPY requirements.txt requirements-api.txt ./
RUN pip install --upgrade pip \
    && pip install -r requirements.txt -r requirements-api.txt

# Install the package.
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-deps -e .

# Model artifacts are expected under /app/artifacts (mount or COPY at build).
COPY artifacts ./artifacts

ENV GPU_POWER_REGISTRY_DIR=/app/artifacts \
    GPU_POWER_MODEL_NAME=random_forest \
    GPU_POWER_MODEL_VERSION=latest

EXPOSE 8000

# Healthcheck hits the readiness endpoint.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status==200 else 1)" || exit 1

CMD ["uvicorn", "gpu_power_pipeline.api:app", "--host", "0.0.0.0", "--port", "8000"]
