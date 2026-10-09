FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    APP_ENV=prod

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

ARG INSTALL_LOCAL_MODELS=false
COPY requirements.txt requirements-local.txt ./
RUN if [ "$INSTALL_LOCAL_MODELS" = "true" ]; then \
        pip install -r requirements-local.txt; \
    else \
        pip install -r requirements.txt; \
    fi

COPY app ./app
COPY scripts/prepare_chunk_tokenizer.py ./scripts/prepare_chunk_tokenizer.py
RUN if [ "$INSTALL_LOCAL_MODELS" != "true" ]; then python scripts/prepare_chunk_tokenizer.py; fi

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
