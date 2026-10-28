# syntax=docker/dockerfile:1

# ---- dashboard: build the React UI -------------------------------------------
FROM node:22-slim AS ui
WORKDIR /build
COPY frontend/package.json frontend/package-lock.json frontend/
RUN cd frontend && npm ci
COPY frontend frontend
COPY eval/results.json eval/results.json
RUN cd frontend && npm run build   # writes /build/app/ui

# ---- builder: install dependencies into a venv -------------------------------
FROM python:3.12-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY requirements.txt .
# CPU-only torch: ~200MB instead of ~2GB of CUDA libraries we'd never use.
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# Bake the embedding model into the image so containers start offline and fast.
ARG EMBEDDING_MODEL=all-mpnet-base-v2
ENV HF_HOME=/opt/hf
RUN python -c "from sentence_transformers import SentenceTransformer as S; S('${EMBEDDING_MODEL}')"

# ---- runtime -----------------------------------------------------------------
FROM python:3.12-slim
ARG EMBEDDING_MODEL=all-mpnet-base-v2
ENV PATH="/opt/venv/bin:$PATH" \
    HF_HOME=/opt/hf \
    HF_HUB_OFFLINE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    EMBEDDING_MODEL=${EMBEDDING_MODEL}

RUN useradd --create-home --uid 10001 app
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder --chown=app /opt/hf /opt/hf
WORKDIR /srv
COPY --chown=app app ./app
COPY --from=ui --chown=app /build/app/ui ./app/ui

USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status != 200)"

# One worker per container: the embedding model is held in memory per process.
# Scale out with more containers sharing Redis rather than more workers.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
