# syntax=docker/dockerfile:1.7

FROM python:3.11-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

COPY requirements.lock pyproject.toml README.md ./
COPY backend/src ./backend/src

RUN python -m venv /opt/ecocompanion-venv \
    && /opt/ecocompanion-venv/bin/pip install \
        --require-hashes \
        --requirement requirements.lock \
    && /opt/ecocompanion-venv/bin/pip install --no-deps .


FROM python:3.11-slim AS runtime

ENV PATH="/opt/ecocompanion-venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="/app/backend/src"

RUN groupadd --gid 1000 ecocompanion \
    && useradd --uid 1000 --gid ecocompanion --create-home ecocompanion

WORKDIR /app

COPY --from=builder /opt/ecocompanion-venv /opt/ecocompanion-venv

RUN mkdir -p /app/artifacts \
    && chown -R ecocompanion:ecocompanion /app

USER ecocompanion

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=3s --start-period=20s --retries=6 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2).read()"]

CMD ["uvicorn", "ecocompanion.api:app", "--host", "0.0.0.0", "--port", "8000"]
